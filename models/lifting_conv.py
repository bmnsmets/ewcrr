import torch
import torch.nn as nn
from multi_conv import MultiConv2d
from torch import Tensor
import math
import torch.nn.functional as F


def rotation_resampling_grid(
    orientations: int, spatial_size: int, align_corners: bool = False
):
    thetas = torch.linspace(0, 2 * torch.pi, orientations + 1)[:orientations]
    cos, sin = thetas.cos(), thetas.sin()

    transforms = torch.zeros((orientations, 2, 3))
    transforms[:, 0, 0] = cos
    transforms[:, 0, 1] = -sin
    transforms[:, 1, 0] = sin
    transforms[:, 1, 1] = cos

    grid = torch.nn.functional.affine_grid(
        transforms,
        [orientations, 1, spatial_size, spatial_size],
        align_corners=align_corners,
    )

    return grid


def rotation_mask(size: int):
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-1, 1, size),
        torch.linspace(-1, 1, size),
        indexing="ij",
    )
    r = torch.sqrt(grid_x.pow(2) + grid_y.pow(2))
    return r <= 1 + 1 / size


def fuse_kernels_2d(ws: list[Tensor] | torch.nn.ParameterList) -> Tensor:
    w = ws[0].transpose(0, 1)  # [C0,C1,H0,W0]
    for v in ws[1:]:  # [Cn+1,Cn,Hn,Wn]
        kh, kw = v.shape[-2:]
        w = F.conv2d(w, v, padding=(kh - 1, kw - 1))
    return w.transpose(0, 1).contiguous()  # [Cout,C0,H,W]


class SE2Lifting(nn.Module):

    def __init__(
        self,
        num_channels: tuple[int, ...] = (1, 64),
        kernel_size: int | tuple[int, ...] = (5,),
        orientations: int = 16,
        device=None,
        dtype=None,
        zero_mean: bool = True,
    ):
        super().__init__()
        assert len(num_channels) > 1
        assert all(n > 0 for n in num_channels)
        assert orientations > 0
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_channels = num_channels
        self.orientations = orientations
        self.zero_mean = zero_mean
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * (len(num_channels) - 1)
        assert len(kernel_size) == len(num_channels) - 1
        assert all(size > 0 for size in kernel_size)
        eff_kernel_size = kernel_size[0]
        for size in kernel_size[1:]:
            eff_kernel_size += 2 * (size // 2)

        self.kernel_size = kernel_size
        self.eff_kernel_size = eff_kernel_size

        grid = rotation_resampling_grid(
            orientations, eff_kernel_size, align_corners=False
        )
        self.register_buffer("grid", grid)

        mask = rotation_mask(eff_kernel_size)
        self.register_buffer("mask", mask)

        self.weights = nn.ParameterList(
            [
                torch.empty(
                    (num_channels[i + 1], num_channels[i], size, size),
                    **factory_kwargs,
                )
                for i, size in enumerate(kernel_size)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for w in self.weights:
            torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5) * 4 / math.pi)
            with torch.no_grad():
                w.mul_(rotation_mask(w.size(-1)))

    def sample_kernels(self) -> Tensor:
        assert isinstance(self.mask, Tensor)
        assert isinstance(self.grid, Tensor)
        w = fuse_kernels_2d(self.weights) * self.mask
        ksize = w.shape[-2:]
        w = w.view(1, -1, *ksize)  # [1,Cn*C0,H,W]
        w = w.expand(self.orientations, -1, -1, -1)  # [NOr,Cn*C0,H,W]
        w = F.grid_sample(w, self.grid, align_corners=False)  # [NOr,Cn*C0,H,W]
        w = w.view(
            self.orientations, self.num_channels[-1], self.num_channels[0], *ksize
        )
        if self.zero_mean:
            w -= torch.mean(w * self.mask, dim=(2, 3, 4)).view(*w.shape[:2], 1, 1, 1)
        return w * self.mask  # [Nor, Cout, Cin, kH, kW]

    def forward(self, x: Tensor):
        w = self.sample_kernels()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    m = SE2Lifting(num_channels=(1, 2, 4, 8), kernel_size=(5, 5, 5), orientations=16)

    with torch.no_grad():
        m.weights[0][0, 0] = 0
        m.weights[0][0, 0, 2, 2:] = 1
        m.weights[1][0, 0] = 0
        m.weights[1][0, 0, 1:3, 1:3] = 1
    k = m.sample_kernels()
    for i in range(m.orientations // 2 + 1):
        f = k[i, 0, 0].detach()
        v = f.abs().max().item()
        plt.imshow(f, cmap="RdBu", vmin=-v, vmax=v)
        plt.colorbar()
        plt.show()
