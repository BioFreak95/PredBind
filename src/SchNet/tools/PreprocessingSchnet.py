import os
import csv
import numpy as np
from ase.io import read
import ase


class PreprocessingSchnet:

    @staticmethod
    def getMolPath(data_path, complex_name, end):
        return data_path + complex_name + '/' + complex_name + end

    @staticmethod
    def getComplexDirNames(data_path):
        complex_names = list()
        for f in os.scandir(data_path):
            if f.is_dir():
                complex_names.append(f.name)
        complex_names.sort()
        return complex_names

    @staticmethod
    def getAllMolPaths(data_path, end):
        complexes = PreprocessingSchnet.getComplexDirNames(data_path)
        ligands = []
        for i in range(len(complexes)):
            ligands.append(PreprocessingSchnet.getMolPath(data_path, complexes[i], end))
        return ligands, complexes

    @staticmethod
    def extractKValue(index_row):
        label = []
        unit = []
        for i in range(len(index_row[4])):
            if (index_row[4][i].isdigit() or index_row[4][i] == '.'):
                label.append(index_row[4][i])
            if (index_row[4][i - 1].isdigit() and index_row[4][i + 1] == 'M'):
                unit.append(index_row[4][i])
        labels = ''.join(str(e) for e in label)
        units = ''.join(str(e) for e in unit)

        if (units == 'm'):
            k = float(labels) / (1e3)
        elif (units == 'u'):
            k = float(labels) / (1e6)
        elif (units == 'n'):
            k = float(labels) / (1e9)
        elif (units == 'p'):
            k = float(labels) / (1e12)
        elif (units == 'f'):
            k = float(labels) / (1e15)
        else:
            raise NotImplementedError
        return k

    @staticmethod
    def calcLabels(complex_data_path, index_path='Data/index/INDEX_refined_data.2016'):
        complexnames = PreprocessingSchnet.getComplexDirNames(complex_data_path)
        with open(index_path) as f:
            reader = csv.reader(f, delimiter='\t')
            data = [(col1)
                    for col1 in reader]
        datas = []
        for i in range(len(data)):
            datas.append(data[i][0].split(' '))
        labels = []
        print(np.shape(np.array(datas)))
        complexes = list(np.array(datas)[:, 0])
        for i in range(len(complexnames)):
            index = complexes.index(complexnames[i])
            label = PreprocessingSchnet.extractKValue(datas[index])
            labels.append(label)
        for i in range(len(labels)):
            labels[i] = -np.log10(labels[i])
        return labels

    @staticmethod
    def getLabels(complex_data_path, index_path='Data/index/INDEX_refined_data.2016'):
        complexnames = PreprocessingSchnet.getComplexDirNames(complex_data_path)
        with open(index_path) as f:
            reader = csv.reader(f, delimiter='\t')
            data = [(col1)
                    for col1 in reader]
        datas = []
        for i in range(len(data)):
            datas.append(data[i][0].split(' '))
        labels = []
        print(np.shape(np.array(datas)))
        complexes = list(np.array(datas)[:, 0])
        for i in range(len(complexnames)):
            index = complexes.index(complexnames[i])
            print(datas[index][3])
            label = np.float(datas[index][3])
            labels.append(label)
        return labels

    @staticmethod
    def createDatabase(database, threshold=20, data_path='../Data/train/',
                       index_path='../Data/index/INDEX_refined_data.2016',
                       ligand_end='_ligand.sdf', alt_ligand_end='_ligand.pdb', prot_end='_pocket.pdb', mode=None,
                       label_type=None, classes=None, n_classes=None, oversample=False, sample_factor=50):

        ligandPaths = PreprocessingSchnet.getAllMolPaths(data_path, ligand_end)
        ligandPaths2 = PreprocessingSchnet.getAllMolPaths(data_path, alt_ligand_end)
        proteinPaths = PreprocessingSchnet.getAllMolPaths(data_path, prot_end)
        labels = PreprocessingSchnet.getLabels(data_path, index_path)

        indexes = np.arange(len(proteinPaths[0]))
        np.random.shuffle(indexes)

        hist = np.histogram(labels, 25)

        for i in indexes:
            atom_list = []
            try:
                atoms2 = read(ligandPaths[0][i], format='sdf')
                for at in atoms2:
                    atom_list.append(at)
                x = atoms2.positions[:, 0].mean()
                y = atoms2.positions[:, 1].mean()
                z = atoms2.positions[:, 2].mean()
            except:
                try:
                    atoms3 = read(ligandPaths2[0][i], format='proteindatabank')
                    x = atoms3.positions[:, 0].mean()
                    y = atoms3.positions[:, 1].mean()
                    z = atoms3.positions[:, 2].mean()
                    for at in atoms3:
                        atom_list.append(at)
                except:
                    print('Does not work')
                    continue

            affi = PreprocessingSchnet.classLabel(labels[i], mode, label_type, classes=classes, n_classes=n_classes,
                                                min_v=np.min(labels), max_v=np.max(labels))

            mean = np.array([x, y, z])

            atoms = read(proteinPaths[0][i], format='proteindatabank')

            for at in atoms:
                dist = np.linalg.norm(at.position - mean)
                if dist <= threshold:
                    atom_list.append(at)

            complexe = [ase.Atoms(atom_list, pbc=(1, 1, 1))]

            if not oversample:
                database.add_systems(complexe, affi)
            else:
                classn = np.zeros(25)
                for j in range(len(hist[1]) - 1):
                    if j == len(hist[1]) - 2:
                        if hist[1][j] <= labels[i] <= hist[1][j + 1]:
                            classn[j] = 1
                    else:
                        if hist[1][j] <= labels[i] < hist[1][j + 1]:
                            classn[j] = 1

                if np.unique(classn, return_counts=True)[1][1] != 1:
                    print('warning -> Onehot is more than one')
                    print(classn)

                ind = np.argmax(classn)
                if hist[0][ind] == 0:
                    print('Warning -> zero-sample')
                    continue

                n_sampling = int(np.ceil((1 / hist[0][ind]) * sample_factor * 25))
                print(i, len(indexes), n_sampling, ind)
                for _ in range(n_sampling):
                    database.add_systems(complexe, affi)

    @staticmethod
    def classLabel(label, mode=None, label_type=None, min_v=None, max_v=None, classes=None, n_classes=1):
        if mode is None:
            affi = [
                {'KD': np.array([label])}
            ]

        elif mode == 'class':
            if classes is not None:
                class_array = classes
            elif classes is None:
                class_array = np.linspace(min_v, max_v, n_classes + 1)
            else:
                raise NotImplementedError

            one_hot = np.zeros(n_classes)
            for i in range(len(class_array) - 1):
                if i == len(class_array) - 2:
                    if label >= class_array[i] and label <= class_array[i + 1]:
                        one_hot[i] = 1
                else:
                    if label >= class_array[i] and label < class_array[i + 1]:
                        one_hot[i] = 1

            if label_type == 'onehot':
                affi = [
                    {'KD': np.array(one_hot)}
                ]
            elif label_type is None:
                affi = [
                    {'KD': np.array([np.argmax(one_hot)], dtype=np.int64)}
                ]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return affi
