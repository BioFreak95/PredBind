from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.tools import voxeldescriptors
from moleculekit.smallmol.smallmol import SmallMol
from .PreprocessingSchnet import PreprocessingSchnet

from htmd.ui import *
import h5py
import numpy as np


# In SchNet, also a set of features can be used. In this work, the features out of the KDEEP-Paper where used.
# They are calculated here in the same way as in the CNN-Part. See src/CNN/tools/Preprocessing

class CreateFeatureset:
    @staticmethod
    # This function contains a bunch of try-catch scenarios. It is the result of long process to reproduce the
    # preprocessing of KDeep with the PDBBind-dataset
    # (Jose Jimenez, KDEEP: Protein–Ligand Absolute Binding Affinity Prediction via 3D-Convolutional Neural Networks)
    # The process is documented in following issues:
    # https://github.com/Acellera/moleculekit/issues/12, https://github.com/Acellera/moleculekit/issues/13 and
    # https://github.com/Acellera/moleculekit/issues/14
    # Feel free to change this part of the code.
    def calcFeatures(number, ligPath, altLigPath, protPath, altProtPath, boxsize, targetpath):
        features = {}
        try:
            sm = SmallMol(ligPath, force_reading=True, fixHs=False)
            x = np.mean(sm.get('coords')[:, 0])
            y = np.mean(sm.get('coords')[:, 1])
            z = np.mean(sm.get('coords')[:, 2])
            smallChannels, sm = voxeldescriptors.getChannels(sm)
        except:
            sm = SmallMol(altLigPath, force_reading=True, fixHs=False)
            x = np.mean(sm.get('coords')[:, 0])
            y = np.mean(sm.get('coords')[:, 1])
            z = np.mean(sm.get('coords')[:, 2])
            smallChannels, sm = voxeldescriptors.getChannels(sm)
        features['smallChannels'] = smallChannels
        features['sm'] = sm
        try:
            prot = Molecule(protPath)
            if prot.numAtoms > 50000:
                factorx = boxsize[0] * 2.5
                factory = boxsize[1] * 2.5
                factorz = boxsize[2] * 2.5
                prot.filter('z < ' + format(z + factorz) + ' and z > ' + format(z - factorz))
                prot.filter('x < ' + format(x + factorx) + ' and x > ' + format(x - factorx))
                prot.filter('y < ' + format(y + factory) + ' and y > ' + format(y - factory))
            prot.filter('protein')
            prot.bonds = prot._getBonds()
            prot = prepareProteinForAtomtyping(prot)
            prot.set(value='Se', field='element', sel='name SE')
            protChannels, prot = voxeldescriptors.getChannels(prot)

        except:
            try:
                prot = Molecule(altProtPath)
                if prot.numAtoms > 50000:
                    factorx = boxsize[0] * 2.5
                    factory = boxsize[1] * 2.5
                    factorz = boxsize[2] * 2.5
                    prot.filter('z < ' + format(z + factorz) + ' and z > ' + format(z - factorz))
                    prot.filter('x < ' + format(x + factorx) + ' and x > ' + format(x - factorx))
                    prot.filter('y < ' + format(y + factory) + ' and y > ' + format(y - factory))
                prot.filter('protein')
                prot.bonds = prot._getBonds()
                prot = prepareProteinForAtomtyping(prot)
                prot.set(value='Se', field='element', sel='name SE')
                protChannels, prot = voxeldescriptors.getChannels(prot)
            except:
                try:
                    prot = Molecule(protPath)
                    if prot.numAtoms > 50000:
                        factorx = boxsize[0] * 2.5
                        factory = boxsize[1] * 2.5
                        factorz = boxsize[2] * 2.5
                        prot.filter('z < ' + format(z + factorz) + ' and z > ' + format(z - factorz))
                        prot.filter('x < ' + format(x + factorx) + ' and x > ' + format(x - factorx))
                        prot.filter('y < ' + format(y + factory) + ' and y > ' + format(y - factory))
                    prot.filter('protein')
                    prot.filter('not resname 3EB')
                    prot = proteinPrepare(prot)
                    prot = autoSegment(prot)
                    # Residues are not supported
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
                    protChannels, prot = voxeldescriptors.getChannels(prot)
                except:
                    f = open("../../Data/prep_log.txt", "a")
                    f.writelines('Protein ' + protPath + ' leads to errors! Proteinnumber: ' + str(number) + '\n')
                    f.close()
                    protChannels = None
        features['protChannels'] = protChannels
        features['prot'] = prot
        return features

    # This function creates a hdf5 file, where all features from all complexes are saved with necessary details.
    @staticmethod
    def createFeatureset(datapath, indexpath, targetpath):
        labels = PreprocessingSchnet.getLabels(datapath, indexpath)
        ligPaths, complexes = PreprocessingSchnet.getAllMolPaths(datapath, '_ligand.mol2')
        protPaths, complexes = PreprocessingSchnet.getAllMolPaths(datapath, '_protein.pdb')
        altLigPaths, complexes = PreprocessingSchnet.getAllMolPaths(datapath, '_ligand.pdb')
        altProtPaths, complexes = PreprocessingSchnet.getAllMolPaths(datapath, '_protein.mol2')

        file = h5py.File(targetpath + 'SchNetTrain00000.hdf5')
        for j in range(len(protPaths)):
            print(j, complexes[j])
            fet = CreateFeatureset.calcFeatures(number=j, ligPath=ligPaths[j],
                                                altLigPath=altLigPaths[j], protPath=protPaths[j],
                                                altProtPath=altProtPaths[j], boxsize=[24, 24, 24],
                                                targetpath=targetpath)
            smallChannels = fet['smallChannels']
            protChannels = fet['protChannels']
            sm = fet['sm']
            prot = fet['prot']

            prot_num = np.array(prot.element, dtype='S')
            lig_num = np.array(sm.get('element'), dtype='S')
            protChannels[:, 7] = 1
            protf = protChannels
            smallChannels[:, 7] = 0
            ligf = smallChannels

            prot_coords_temp = prot.coords
            lig_coords_temp = sm.get('coords')
            prot_coords = []
            lig_coords = []
            for i in range(len(prot_coords_temp)):
                coord = np.array([prot_coords_temp[i][0][0], prot_coords_temp[i][1][0], prot_coords_temp[i][2][0]])
                prot_coords.append(coord)

            for i in range(len(lig_coords_temp)):
                coord = np.array([lig_coords_temp[i][0][0], lig_coords_temp[i][1][0], lig_coords_temp[i][2][0]])
                lig_coords.append(coord)

            if (j % 50 == 0):
                file.close()
                file = h5py.File(targetpath + 'SchNetTrain' + "{0:0=5d}".format(j) + '.hdf5')
            file[str(j) + '/prot'] = protf
            file[str(j) + '/lig'] = ligf
            file[str(j) + '/protnum'] = prot_num
            file[str(j) + '/lignum'] = lig_num
            file[str(j) + '/ligcoords'] = lig_coords
            file[str(j) + '/protcoords'] = prot_coords
            file[str(j) + '/Complex'] = complexes[j]
            file[str(j) + '/label'] = labels[j]
            print(j)
        file.close()
