from ..tools.PreprocessingSchnet import PreprocessingSchnet
import numpy as np
import torch


class ShiftedSigmoid:
    def __init__(self):
        labels = PreprocessingSchnet.getLabels('../Data/train', '../Data/index/INDEX_refined_data.2016')
        self.maxv = np.max(labels)
        self.minv = np.min(labels)
    
    def forward(self,input):
        return (self.maxv - self.minv) * torch.sigmoid(input) + self.minv

