import torch
import torch.nn as nn
from torch import Tensor
import math
import torch.nn.functional as F
from typing import Literal


def rotation_resampling_grid(
    orientations: int, spatial_size: int, align_corners: bool = False
):
    thetas = torch.linspace(0, 2 * torch.pi, orientations + 1)[:orientations]
    cos, sin = thetas.cos(), thetas.sin()

    transforms = torch.zeros((orientations, 2, 3))
    transforms[:, 0, 0] = -cos
    transforms[:, 0, 1] = -sin
    transforms[:, 1, 0] = sin
    transforms[:, 1, 1] = -cos

    grid = torch.nn.functional.affine_grid(
        transforms,
        [orientations, 1, spatial_size, spatial_size],
        align_corners=align_corners,
    )

    return grid


def rotation_mask(size: int) -> Tensor:
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-1, 1, size),
        torch.linspace(-1, 1, size),
        indexing="ij",
    )
    r = torch.sqrt(grid_x.pow(2) + grid_y.pow(2))
    return r <= 1 + 1 / size


class SE2LiftMultiConv2d(nn.Module):

    def __init__(
        self,
        num_channels: tuple[int, ...] = (1, 64),
        size_kernels: int | tuple[int, ...] = (5,),
        orientations: int = 16,
        device=None,
        dtype=None,
        zero_mean: bool = True,
        sn_size: int = 256,
    ):
        super().__init__()
        assert len(num_channels) > 1
        assert all(n > 0 for n in num_channels)
        assert orientations > 0
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_channels = num_channels
        self.orientations = orientations
        self.zero_mean = zero_mean
        self.sn_size = sn_size
        if isinstance(size_kernels, int):
            size_kernels = (size_kernels,) * (len(num_channels) - 1)
        assert len(size_kernels) == len(num_channels) - 1
        assert all(size > 0 for size in size_kernels)
        eff_kernel_size = size_kernels[0]
        for size in size_kernels[1:]:
            eff_kernel_size += 2 * (size // 2)

        self.size_kernels = size_kernels
        self.eff_kernel_size = eff_kernel_size
        self.padding_total = sum(size // 2 for size in size_kernels)

        # grid locations for resampling rotated version of the mother kernels
        grid = rotation_resampling_grid(
            orientations, eff_kernel_size, align_corners=False
        ).to(device)
        self.register_buffer("grid", grid)

        # mask to make the effective kernels supported in a circle
        mask = rotation_mask(eff_kernel_size).to(device)
        self.register_buffer("mask", mask)

        # caching the spectral norm estimate
        L = torch.ones(1, requires_grad=True, device=device)
        self.register_buffer("L", L)

        self.weights = nn.ParameterList(
            [
                torch.empty(
                    (num_channels[i + 1], num_channels[i], size, size),
                    **factory_kwargs,
                )
                for i, size in enumerate(size_kernels)
            ]
        )

        dsize = eff_kernel_size + 2 * (eff_kernel_size // 2)
        dirac = torch.zeros((1, 1, dsize, dsize), **factory_kwargs)
        dirac[0, 0, dsize // 2, dsize // 2] = 1
        self.register_buffer("dirac", dirac)

        self.reset_parameters()

    @property
    def device(self):
        return self.weights[0].device

    @property
    def dtype(self):
        return self.weights[0].device

    def reset_parameters(self) -> None:
        for w in self.weights:
            torch.nn.init.kaiming_uniform_(w, a=math.sqrt(4))
            with torch.no_grad():
                w.mul_(rotation_mask(w.size(-1)).to(w.device))

    def num_parameters(self) -> int:
        n = 0
        for w, sz in zip(self.weights, self.size_kernels):
            n += w.numel() * rotation_mask(sz).sum().item() // (sz**2)
        return n

    def _fuse_2d_filters(self) -> Tensor:
        assert isinstance(self.dirac, Tensor) and isinstance(self.mask, Tensor)
        filters = self.dirac
        ws = [w * rotation_mask(w.size(-1)).to(w.device) for w in self.weights]
        if self.zero_mean:
            ws0 = ws[0]  # [C1,C0,H,W]
            mean0 = torch.sum(ws0, dim=(1, 2, 3)) / rotation_mask(
                ws0.size(-1)
            ).sum().to(
                ws0.device
            )  # [C1]
            ws0 = ws0 - mean0.view(ws0.size(0), 1, 1, 1)
            ws = [ws0 * rotation_mask(ws0.size(-1)).to(ws0.device), *ws[1:]]
        for w in ws:
            filters = F.conv2d(filters, w, bias=None)
        return filters * self.mask

    def _sample_filter_stack(self) -> Tensor:
        assert isinstance(self.mask, Tensor)
        assert isinstance(self.grid, Tensor)
        w = self._fuse_2d_filters()
        ksize = w.shape[-2:]
        w = w.view(1, -1, *ksize)  # [1,Cn*C0,H,W]
        w = w.expand(self.orientations, -1, -1, -1)  # [NOr,Cn*C0,H,W]
        w = F.grid_sample(w, self.grid, align_corners=False)  # [NOr,Cn*C0,H,W]
        w = w.view(
            self.orientations, self.num_channels[-1], self.num_channels[0], *ksize
        )
        return w * self.mask  # [Nor, Cout, Cin, kH, kW]

    def get_filters(self) -> Tensor:
        assert isinstance(self.dirac, Tensor)
        i, j = self.padding_total, self.padding_total + self.eff_kernel_size
        return self.convolution(self.dirac)[:, :, 0, i:j, i:j]

    def forward(self, x: Tensor):
        return self.convolution(x)

    def convolution(self, x: Tensor):
        assert isinstance(self.L, Tensor)
        x = x / torch.sqrt(self.L)  # [B,Cin,H,W]
        w = self._sample_filter_stack()  # [NOr,Cout,Cin,kH,kH]
        w = w.transpose(0, 1)  # [Cout,NOr,Cin,kH,kH]
        w = w.reshape(-1, *w.shape[-3:])  # [Cout*NOr,Cin,kH,kH]
        y = F.conv2d(
            x, w, bias=None, padding=(self.eff_kernel_size // 2)
        )  # [B,Cout*NOr,H,W]
        y = y.reshape(x.size(0), -1, self.orientations, *y.shape[-2:])
        return y  # [B,Cout,NOr,H,W]

    def transpose(self, x: Tensor):
        assert isinstance(self.L, Tensor)
        x = x / torch.sqrt(self.L)  # [B,Cout,NOr,H,W]
        x = x.reshape(x.size(0), -1, x.size(-2), x.size(-1))  # [B,Cout*NOr,H,W]
        w = self._sample_filter_stack()  # [NOr,Cout,Cin,kH,kH]
        w = w.transpose(0, 1)  # [NCout,NOr,Cin,kH,kH]
        w = w.reshape(-1, *w.shape[-3:])  # [Cout*NOr,Cin,kH,kH]
        y = F.conv_transpose2d(x, w, bias=None, padding=(self.eff_kernel_size // 2))
        return y  # [B,Cin,H,W]

    def get_kernel_WtW(self):
        assert isinstance(self.dirac, Tensor)
        return self.transpose(self.convolution(self.dirac))

    def spectrum(self):
        kernel = self.get_kernel_WtW()
        padding = (self.sn_size - 1) // 2 - self.padding_total
        return torch.fft.fft2(F.pad(kernel, (padding,) * 4))

    def spectral_norm(
        self,
        mode: Literal["Fourier"] | Literal["power_method"] = "Fourier",
        n_steps: int = 1000,
    ) -> Tensor:
        if mode == "Fourier":
            self.L = torch.ones(1, device=self.device)
            self.L = self.spectrum().abs().max()
        elif mode == "power_method":
            self.L = torch.ones(1, device=self.device)
            u = torch.randn((1, 1, self.sn_size, self.sn_size), device=self.device)
            with torch.no_grad():
                for _ in range(n_steps):
                    u = self.transpose(self.convolution(u))
                    u = u / torch.linalg.norm(u)
            self.L = torch.linalg.norm(self.transpose(self.convolution(u)))

        return self.L

    def __repr__(self):
        s1 = f"num_channels={self.num_channels}"
        s2 = f"kernel_size={self.size_kernels}"
        s3 = f"orientations={self.orientations}"
        return f"{self.__class__.__name__}({s1}, {s2}, {s3})"


def number_kernel_params(kernel_sizes: tuple[int, ...]) -> int:
    return sum(int(rotation_mask(sz).sum().item()) for sz in kernel_sizes)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    m = SE2LiftMultiConv2d(
        num_channels=(1, 4, 8, 60), size_kernels=(5, 5, 5), orientations=1
    )

    # with torch.no_grad():
    #     m.weights[0][0, 0] = 0
    #     m.weights[0][0, 0, 2, 2:] = 1
    #     m.weights[1][0, 0] = 0
    #     m.weights[1][0, 0, 1:3, 1:3] = 1
    k = m._sample_filter_stack()
    from multi_conv import MultiConv2d

    n = MultiConv2d(num_channels=(1, 4, 8, 60), size_kernels=(5, 5, 5))
    # for i in range(m.orientations // 2 + 1):
    #     f = k[i, 0, 0].detach()
    #     v = f.abs().max().item()
    #     plt.imshow(f, cmap="RdBu", vmin=-v, vmax=v)
    #     plt.colorbar()
    #     plt.show()

    for pm, pn in zip(m.parameters(), n.parameters()):
        with torch.no_grad():
            pn.copy_(pm)

    k1 = m.get_filters().detach()[0, 0]
    k2 = n.get_filters().detach()[0, 0]
    v = max(k1.abs().max().item(), k2.abs().max().item())
    plt.imshow(k1, cmap="RdBu", vmin=-v, vmax=v)
    plt.colorbar()
    plt.show()
    plt.imshow(k2, cmap="RdBu", vmin=-v, vmax=v)
    plt.colorbar()
    plt.show()
