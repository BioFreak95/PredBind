from torch.utils.data.dataset import Dataset
from src.CNN.tools.Rotations import Rotations
import numpy as np
import h5py


class OwnDataset(Dataset):
    def __init__(self, indices, path):
        self.hdf5file = path
        self.rot = Rotations()
        self.indices = np.tile(indices, 24)
        self.length = len(indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        with h5py.File(self.hdf5file, 'r') as file:
            data = self.rot.rotation(
                data=file[str(data_idx) + '/data'][()][0],
                k=np.floor_divide(index, self.length)).copy()

            label = file[str(data_idx) + '/label'][()]

        return data, label

    def __len__(self):
        return self.length * 24
