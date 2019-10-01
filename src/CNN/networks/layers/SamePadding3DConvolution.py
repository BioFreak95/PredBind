# like mentioned in issue #3867 of pytorch (https://github.com/pytorch/pytorch/issues/3867)
# -> no same padding in convolution in pytorch -> This implementation has the same behavior
# like the same padding convolution in tensorflow


import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SamePadding3DConvolution(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            bias=True,
            stride=1,
            dilation=1,
            groups=1
    ):
        super(SamePadding3DConvolution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            groups=groups
        )
        nn.init.xavier_uniform_(self.conv.weight)
        self.conv.bias.data.zero_()

    def forward(self, data):
        batch_size, channels, height, width, depth = np.shape(data)

        # Output-size out = in / stride
        height2 = np.ceil(height / self.stride)
        width2 = np.ceil(width / self.stride)
        depth2 = np.ceil(depth / self.stride)

        # converted formular of convolution (pytorch documentation (conv3d))
        ph = np.int(
            (height2 - 1) * self.stride +
            (self.kernel_size - 1) * self.dilation + 1 -
            height
        )
        pw = np.int(
            (width2 - 1) * self.stride +
            (self.kernel_size - 1) * self.dilation + 1 -
            width
        )
        pd = np.int(
            (depth2 - 1) * self.stride +
            (self.kernel_size - 1) * self.dilation + 1 -
            depth
        )

        # calculate padded data before convolution
        padded_data = F.pad(
            data,
            (
                ph // 2,
                ph - ph // 2,
                pw // 2,
                pw - pw // 2,
                pd // 2,
                pd - pd // 2
            )
        )

        output = self.conv(padded_data)

        return output
