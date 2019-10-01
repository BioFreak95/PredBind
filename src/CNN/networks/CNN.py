import torch
import torch.nn as nn
import torch.nn.functional as F
from src.CNN.networks.layers.FireModule import FireModule


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv3d(16, 96, 3, stride=2, padding=1)
        self.fire1 = FireModule(96, 16)
        self.fire2 = FireModule(128, 16)
        self.fire3 = FireModule(128, 32)
        self.maxPool1 = nn.MaxPool3d(3, 2)
        self.fire4 = FireModule(256, 32)
        self.fire5 = FireModule(256, 48)
        self.fire6 = FireModule(384, 48)
        self.fire7 = FireModule(384, 64)
        self.avPool1 = nn.AvgPool3d(3, 2)
        self.dense = nn.Linear(4096, 1)

        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1.bias.data.zero_()

    def forward(self, data):
        conv1 = F.relu(self.conv1(data))
        fire1 = self.fire1(conv1)
        fire2 = self.fire2(fire1)
        fire3 = self.fire3(fire2)
        maxpool1 = F.relu(self.maxPool1(fire3))
        fire4 = self.fire4(maxpool1)
        fire5 = self.fire5(fire4)
        fire6 = self.fire6(fire5)
        fire7 = self.fire7(fire6)
        av1 = F.relu(self.avPool1(fire7))
        flat = av1.view(-1, 4096)
        out = self.dense(flat)

        return out
