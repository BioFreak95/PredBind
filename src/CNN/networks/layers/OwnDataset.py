from torch.utils.data.dataset import Dataset
from src.CNN.tools.Rotations import Rotations
import numpy as np
import h5py


class OwnDataset(Dataset):
    def __init__(self, indices, path, rotations=True):
        self.hdf5file = path
        self.rot = Rotations()
        self.rotations = rotations
        if self.rotations:
            self.indices = np.tile(indices, 24)
        else:
            self.indices = indices
        self.length = len(indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        with h5py.File(self.hdf5file, 'r') as file:
            if self.rotations:
                data = self.rot.rotation(
                    data=file[str(data_idx) + '/data'][()][0],
                    k=np.floor_divide(index, self.length)).copy()
            else:
                data = file[str(data_idx) + '/data'][()][0]
            label = file[str(data_idx) + '/label'][()]
            label = -np.log10(np.exp(-label))

        return data, label

    def __len__(self):
        if self.rotations:
            return self.length * 24
        else:
            return self.length
