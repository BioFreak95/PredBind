from src.SchNet.tools.preprocessing_schnet import preprocessing_schnet
import numpy as np
import torch


class ShiftedSigmoid:
    def __init__(self):
        labels = preprocessing_schnet.get_labels('../Data/train', '../Data/index/INDEX_refined_data.2016')
        self.maxv = np.max(labels)
        self.minv = np.min(labels)
    
    def forward(self,input):
        return (self.maxv - self.minv) * torch.sigmoid(input) + self.minv

