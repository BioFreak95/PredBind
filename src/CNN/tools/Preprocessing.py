from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.tools import voxeldescriptors
from moleculekit.smallmol.smallmol import SmallMol

from htmd.ui import *
import pybel

import h5py
import numpy as np
import csv
import os


class Preprocessing:

    def __init__(self, boxsize=None):
        if boxsize is None:
            boxsize = [24, 24, 24]
        self.boxsize = list(boxsize)

    def calcDatasetVoxel(self, protPath, ligPath, number, altProtPath, altLigPath):
        dataset = list()
        print(ligPath)
        try:
            sm = SmallMol(ligPath, force_reading=True)
            x = np.mean(sm.get('coords')[:, 0])
            y = np.mean(sm.get('coords')[:, 1])
            z = np.mean(sm.get('coords')[:, 2])
            fs, cs, ns = voxeldescriptors.getVoxelDescriptors(
                sm,
                center=[x, y, z],
                boxsize=self.boxsize
            )
        except:
            sm = SmallMol(altLigPath, force_reading=True)
            x = np.mean(sm.get('coords')[:, 0])
            y = np.mean(sm.get('coords')[:, 1])
            z = np.mean(sm.get('coords')[:, 2])
            fs, cs, ns = voxeldescriptors.getVoxelDescriptors(
                sm,
                center=[x, y, z],
                boxsize=self.boxsize
            )
        f, c, n = self.calcProtVoxel(x, y, z, protPath, number, altProtPath)
        feature_protein = f
        feature_protein_shaped = f.reshape(n[0], n[1], n[2], f.shape[1])
        feature_ligand = fs
        feature_ligand_shaped = fs.reshape(ns[0], ns[1], ns[2], fs.shape[1])
        datapoint = np.concatenate((feature_protein_shaped, feature_ligand_shaped), axis=3).transpose([3, 0, 1, 2])
        dataset.append(datapoint)

        return np.array(dataset), np.array(c), np.array(feature_protein), np.array(feature_ligand), np.array(
            feature_protein_shaped), np.array(feature_ligand_shaped)

    # This function contains a bunch of try-catch scenarios. It is the result of long process to reproduce the
    # preprocessing of KDeep witth the PDBBind-dataset
    # (Jose Jimenez, KDEEP: Protein–Ligand Absolute Binding Affinity Prediction via 3D-Convolutional Neural Networks)
    # The process is documented in following issues:
    # https://github.com/Acellera/moleculekit/issues/12, https://github.com/Acellera/moleculekit/issues/13 and
    # https://github.com/Acellera/moleculekit/issues/14
    # Feel free to change this part of the code.
    def calcProtVoxel(self, x, y, z, protPath, number, altProtPath):
        try:
            prot = Molecule(protPath)
            if prot.numAtoms > 50000:
                factorx = self.boxsize[0] * 2.5
                factory = self.boxsize[1] * 2.5
                factorz = self.boxsize[2] * 2.5
                prot.filter('z < ' + format(z + factorz) + ' and z > ' + format(z - factorz))
                prot.filter('x < ' + format(x + factorx) + ' and x > ' + format(x - factorx))
                prot.filter('y < ' + format(y + factory) + ' and y > ' + format(y - factory))
            prot.filter('protein')
            prot.bonds = prot._getBonds()
            prot = prepareProteinForAtomtyping(prot)
            prot.set(value='Se', field='element', sel='name SE')
            f, c, n = voxeldescriptors.getVoxelDescriptors(
                prot,
                center=[x, y, z],
                boxsize=self.boxsize
            )

        except:
            try:
                prot = Molecule(protPath)
                if prot.numAtoms > 50000:
                    factorx = self.boxsize[0] * 2.5
                    factory = self.boxsize[1] * 2.5
                    factorz = self.boxsize[2] * 2.5
                    prot.filter('z < ' + format(z + factorz) + ' and z > ' + format(z - factorz))
                    prot.filter('x < ' + format(x + factorx) + ' and x > ' + format(x - factorx))
                    prot.filter('y < ' + format(y + factory) + ' and y > ' + format(y - factory))
                prot.filter('protein')
                prot = proteinPrepare(prot)
                prot = autoSegment(prot)
                prot = charmm.build(prot, ionize=False)
                f, c, n = voxeldescriptors.getVoxelDescriptors(
                    prot,
                    center=[x, y, z],
                    boxsize=self.boxsize
                )
            except:
                try:
                    prot = Molecule(altProtPath)
                    if prot.numAtoms > 50000:
                        factorx = self.boxsize[0] * 2.5
                        factory = self.boxsize[1] * 2.5
                        factorz = self.boxsize[2] * 2.5
                        prot.filter('z < ' + format(z + factorz) + ' and z > ' + format(z - factorz))
                        prot.filter('x < ' + format(x + factorx) + ' and x > ' + format(x - factorx))
                        prot.filter('y < ' + format(y + factory) + ' and y > ' + format(y - factory))
                    prot.filter('protein')
                    prot.bonds = prot._getBonds()
                    prot = prepareProteinForAtomtyping(prot)
                    prot.set(value='Se', field='element', sel='name SE')
                    f, c, n = voxeldescriptors.getVoxelDescriptors(
                        prot,
                        center=[x, y, z],
                        boxsize=self.boxsize
                    )
                except:
                    try:
                        prot = Molecule(protPath)
                        if prot.numAtoms > 30000:
                            factorx = self.boxsize[0] * 2.5
                            factory = self.boxsize[1] * 2.5
                            factorz = self.boxsize[2] * 2.5
                            prot.filter('z < ' + format(z + factorz) + ' and z > ' + format(z - factorz))
                            prot.filter('x < ' + format(x + factorx) + ' and x > ' + format(x - factorx))
                            prot.filter('y < ' + format(y + factory) + ' and y > ' + format(y - factory))
                        prot.filter('protein')
                        prot = proteinPrepare(prot)
                        prot = autoSegment(prot)
                        try:
                            prot.mutateResidue('resname TPO', 'THR')
                        except:
                            pass
                        try:
                            prot.mutateResidue('resname MSE', 'MET')
                        except:
                            pass
                        try:
                            prot.mutateResidue('resname SEP', 'SER')
                        except:
                            pass
                        prot = charmm.build(prot, ionize=False)
                        f, c, n = voxeldescriptors.getVoxelDescriptors(
                            prot,
                            center=[x, y, z],
                            boxsize=self.boxsize
                        )
                    except:
                        f = open("../../Data/prep_log.txt", "a")
                        f.writelines('Protein ' + protPath + ' leads to errors! Proteinnumber: ' + str(number) + '\n')
                        f.close()
                        f = np.random.rand(13824, 8)
                        c = np.random.rand(13824, 3)
                        n = [24, 24, 24]
        return f, c, n

    @staticmethod
    def getComplexDirNames(data_path):
        complex_names = list()
        for f in os.scandir(data_path):
            if f.is_dir():
                complex_names.append(f.name)
        complex_names.sort()
        return complex_names

    @staticmethod
    def writeDataToFile(data, datapath):
        file = h5py.File(datapath)
        for i in range(len(data)):
            file[format(i)] = data[i]

    @staticmethod
    def removeFolder(path):
        import shutil
        shutil.rmtree(path)

    @staticmethod
    def removeFile(path):
        import os
        os.remove(path)

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
    def getLabels(complex_data_path, index_path='Data/index/INDEX_refined_data.2016', complexnames=None):
        if complexnames is None:
            complexnames = Preprocessing.getComplexDirNames(complex_data_path)
        with open(index_path) as f:
            reader = csv.reader(f, delimiter='\t')
            data = [(col1)
                    for col1 in reader]
        datas = []
        for i in range(len(data)):
            datas.append(data[i][0].split(' '))
        labels = []
        complexes = list(np.array(datas)[:, 0])
        for i in range(len(complexnames)):
            index = complexes.index(complexnames[i])
            label = Preprocessing.extractKValue(datas[index])
            labels.append(label)
        for i in range(len(labels)):
            labels[i] = -np.log10(labels[i])
        return labels

    @staticmethod
    def getMolPath(data_path, complex_name, namespace):
        return data_path + complex_name + '/' + complex_name + namespace

    @staticmethod
    def getAllMolPaths(data_path, namespace, complexes=None):
        if complexes is None:
            complexes = Preprocessing.getComplexDirNames(data_path)
        ligands = []
        for i in range(len(complexes)):
            ligands.append(Preprocessing.getMolPath(data_path, complexes[i], namespace))
        return ligands, complexes

    def getAllVoxelisedData(self, data_path, protNamespace, ligNamespace):
        protPaths = Preprocessing.getAllMolPaths(data_path, protNamespace)
        ligPaths = Preprocessing.getAllMolPaths(data_path, ligNamespace)
        dataset = []
        for i in range(len(ligPaths)):
            print(i + 1, 'of', len(ligPaths))
            dataset.append(self.calcDatasetVoxel(protPaths[i], ligPaths[i], i)[0])
        return dataset

    def createVoxelisedFile(self, datapath, savepointnum, protNamespace, altProNamespace, altLigNamespace, ligNamespace,
                            namespace='../Data/HDF5/data', startpoint=0, complexes=None):
        j = 0
        print(complexes)
        protPaths, c = Preprocessing.getAllMolPaths(datapath, protNamespace, complexes=complexes)
        altprotPaths, c = Preprocessing.getAllMolPaths(datapath, altProNamespace, complexes=complexes)
        ligPaths, c = Preprocessing.getAllMolPaths(datapath, ligNamespace, complexes=complexes)
        altLigPaths, c = Preprocessing.getAllMolPaths(datapath, altLigNamespace, complexes=complexes)
        file = h5py.File(namespace + "{0:0=5d}".format(0) + '.hdf5')
        for i in range(startpoint, len(ligPaths)):
            if (i % savepointnum == 0):
                file.close()
                file = h5py.File(namespace + "{0:0=5d}".format(i) + '.hdf5')
                j = 0
            data = self.calcDatasetVoxel(protPath=protPaths[i], ligPath=ligPaths[i], number=i,
                                         altProtPath=altprotPaths[i], altLigPath=altLigPaths[i])[0]
            file[format(j)] = data
            print(i, 'of', len(ligPaths))
            j += 1
        file.close()

    @staticmethod
    def convertData(namespace, startExt, targetExt, proteinnumber, path):
        paths, complexes = Preprocessing.getAllMolPaths(path, namespace + '.' + startExt)
        w = pybel.readfile(startExt, paths[proteinnumber])
        molec = next(w)
        out = pybel.Outputfile(targetExt, path + complexes[proteinnumber] + '/' + complexes[proteinnumber] +
                               namespace + '.' + targetExt)
        out.write(molec)
        out.close()
