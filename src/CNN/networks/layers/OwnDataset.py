from torch.utils.data.dataset import Dataset
from src.CNN.tools.Rotations import Rotations
import numpy as np
import h5py


# Creates a own Dataloader for pytorch
class OwnDataset(Dataset):
    def __init__(self, indices, path, rotations=True, version=2):
        self.hdf5file = path
        self.rot = Rotations()
        self.rotations = rotations
        if self.rotations:
            self.indices = np.tile(indices, 24)
        else:
            self.indices = indices
        self.length = len(indices)
        self.version = version

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

            # In the old version of the code, the labels were calculated with the natural logarithm instead with the log10. So this is the recalc
            if self.version == 1:
                label = -np.log10(np.exp(-label))

        return data, label

    def __len__(self):
        if self.rotations:
            return self.length * 24
        else:
            return self.length
