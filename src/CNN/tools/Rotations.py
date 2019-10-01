import numpy as np


class Rotations:

    def __init__(self):
        self.rotations = [
            lambda data: data,
            lambda data: np.rot90(data, axes=(2, 1)),
            lambda data: np.rot90(np.rot90(data, axes=(2, 1)), axes=(2, 3)),
            lambda data: np.flip(np.rot90(data, axes=(2, 1)), axis=(2, 3)),
            lambda data: np.rot90(np.rot90(data, axes=(2, 1)), axes=(3, 2)),
            lambda data: np.flip(data, axis=(2, 1)),
            lambda data: np.rot90(np.flip(data, axis=(2, 1)), axes=(2, 3)),
            lambda data: np.flip(data, axis=(3, 1)),
            lambda data: np.rot90(np.flip(data, axis=(2, 1)), axes=(3, 2)),
            lambda data: np.rot90(data, axes=(1, 2)),
            lambda data: np.rot90(np.rot90(data, axes=(1, 2)), axes=(2, 3)),
            lambda data: np.flip(np.rot90(data, axes=(1, 2)), axis=(2, 3)),
            lambda data: np.rot90(np.rot90(data, axes=(1, 2)), axes=(3, 2)),
            lambda data: np.rot90(data, axes=(2, 3)),
            lambda data: np.flip(data, axis=(2, 3)),
            lambda data: np.rot90(data, axes=(3, 2)),
            lambda data: np.rot90(np.rot90(data, axes=(2, 1)), axes=(3, 1)),
            lambda data: np.rot90(np.rot90(data, axes=(3, 2)), axes=(1, 2)),
            lambda data: np.rot90(np.flip(data, axis=(2, 1)), axes=(3, 1)),
            lambda data: np.rot90(data, axes=(1, 3)),
            lambda data: np.rot90(np.rot90(data, axes=(1, 2)), axes=(3, 1)),
            lambda data: np.rot90(np.rot90(data, axes=(2, 1)), axes=(1, 3)),
            lambda data: np.rot90(data, axes=(3, 1)),
            lambda data: np.rot90(np.flip(data, axis=(2, 3)), axes=(3, 1))
        ]

    # Calculates all 24 different permutations of an cubic tensor shape have to be (Channel, X, Y, Z)
    # calcRotation2 is fastest method to calculate all 24 permuations at once.
    @staticmethod
    def calcAllRotationsFast(data):
        t1 = []
        n1 = data
        for i in range(4):
            n1 = np.rot90(n1, axes=(2, 1))
            t1.append(n1)
            n2 = np.copy(n1)
            for j in range(3):
                n2 = np.rot90(n2, axes=(2, 3))
                t1.append(n2)
        for k in [0, 2, 4, 6, 8, 10, 12, 14]:
            t1.append(np.rot90(t1[k], axes=(3, 1)))
        return np.array(t1)

    # Calculates all 24 different permutations of an cubic tensor for a whole batch
    # Shape have to be (Complex-Number, Channel, X, Y, Z)
    @staticmethod
    def calcRotationsBatch(data):
        dataset = None
        for i in range(data.shape[0]):
            if dataset is None:
                dataset = Rotations.calcAllRotationsFast(data[i])
            else:
                dataset = np.append(dataset, Rotations.calcAllRotationsFast(data[i]), axis=0)
        return dataset

    # Is a little bit slower than calcAllRotationsFast, but uses the rotation function.
    @staticmethod
    def calcAllRotations(self, data):
        t = []
        for i in range(24):
            t.append(self.rotation(data, i))
        return t

    # calculate one specific rotation by index. The rotation is specified by the function-list "rotations".
    def rotation(self, data, k):
        return self.rotations[k](data)
