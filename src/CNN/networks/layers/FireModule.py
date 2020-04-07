import torch
import torch.nn.functional as F
import torch.nn as nn


# creates the fireModule. See KDeep-Paper
class FireModule(torch.nn.Module):

    def __init__(self, in_channels, n_squeeze_filter):
        super(FireModule, self).__init__()

        self.squeeze = nn.Conv3d(
            in_channels,
            n_squeeze_filter,
            kernel_size=1
        )
        self.expand_1 = nn.Conv3d(
            n_squeeze_filter,
            4 * n_squeeze_filter,
            kernel_size=1
        )
        self.expand_3 = nn.Conv3d(
            n_squeeze_filter,
            4 * n_squeeze_filter,
            kernel_size=3,
            padding=1
        )

        nn.init.xavier_uniform_(self.squeeze.weight)
        self.squeeze.bias.data.zero_()
        nn.init.xavier_uniform_(self.expand_1.weight)
        self.expand_1.bias.data.zero_()
        nn.init.xavier_uniform_(self.expand_3.weight)
        self.expand_3.bias.data.zero_()

    def forward(self, data):
        squeeze = F.relu(self.squeeze(data))
        expand_1 = F.relu(self.expand_1(squeeze))
        expand_3 = F.relu(self.expand_3(squeeze))
        output = torch.cat([expand_1, expand_3], 1)

        return output
