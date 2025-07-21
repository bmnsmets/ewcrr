import torch
import torch.nn as nn
from multi_conv import MultiConv2d


class SE2Lifting(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        orientations: int = 8,
        kernel_size: int = 7,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.orientations = orientations

        self.weights = nn.Parameter(
            torch.empty(
                (out_channels, in_channels, kernel_size, kernel_size), **factory_kwargs
            )
        )

        self.mask = torch.zeros((kernel_size, kernel_size), **factory_kwargs)


class LiftingMultiConv2d(MultiConv2d):

    def __init__(
        self,
        in_channels: int = 1,
        r2_channels: int = 64,
        r2_kernel_sizes: tuple[int, ...] = (5, 5, 5),
        se2_channels: int | None = None,
        se2_kernel_size: tuple[int, ...] = (5, 5, 5),
        se2_orientations: int = 8,
        zero_mean: bool = True,
        sn_size: int = 256,
    ):
        super().__init__()
