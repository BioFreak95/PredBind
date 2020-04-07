from src.CNN.tools.Preprocessing import *
import os

# The preprocessing was a reproduction of the KDeep-paper
# (Jose Jimenez, KDEEP: Protein–Ligand Absolute Binding Affinity Prediction via 3D-Convolutional Neural Networks)
# There where some problems in the preprocessing which are discussed in following issues:
# https://github.com/Acellera/moleculekit/issues/12, https://github.com/Acellera/moleculekit/issues/13 and
# https://github.com/Acellera/moleculekit/issues/14
# This resulting script was the result and worked for me.
# NOTE: In PDBBIND 4bcn and 4bcp have the residue TPO and 4n6z has residue SEP -> Both can not read by this package
# -> What you can do is, delete this manually or reduce the cutoff in the voxelisation
# For failing proteins (3 for PDBBIND2016) -> Random voxels are created and mentioned in the log-file
# -> You can delete them if you like or try to fix them manually.
# Recommended is, that you mutate TPO to THR and SEP to SER -> An exception to fix this, if necessary is implemented

# This script expect the train and test data from the pdbbind-dataset in the ../Data-dict

prep = Preprocessing()

# The database is created step by step and is very expensive. Because sometimes the Data is not readable,
# this can most of the time be fixed by convert the pdb to mol2 and use this instead. -> Make this only if necessary
# -> converting is expensive and can crash the system for big molecules

trainPath = '../../Data/train/'
protNamespace = '_protein'
paths, complexes = Preprocessing.getAllMolPaths(trainPath, protNamespace + '.pdb')
for i in range(len(paths)):
    Preprocessing.convertData(namespace=protNamespace, startExt='pdb', targetExt='mol2', proteinnumber=i,
                              path=trainPath)

# Try first with pdb if this not work, change it to mol2 -> The reason is, that in pdbbind the data comes
# from experiments and can sometimes a little bit broken -> In this case it can not be readed -> converting can help

prep.createVoxelisedFile(datapath='../../Data/train/', savepointnum=50, protNamespace='_protein.pdb',
                         ligNamespace='_ligand.mol2', namespace='../../Data/temp/train/train',
                         altProNamespace='_protein.mol2', altLigNamespace='_ligand.pdb', startpoint=0)

# Repeat the same for test-data
testPath = '../../Data/test/'
protNamespace = '_protein'
paths, complexes = Preprocessing.getAllMolPaths(trainPath, protNamespace + '.pdb')
for i in range(len(paths)):
    Preprocessing.convertData(namespace=protNamespace, startExt='pdb', targetExt='mol2', proteinnumber=i, path=testPath)

prep.createVoxelisedFile(datapath='../../Data/test/', savepointnum=50, protNamespace='_protein.pdb',
                         ligNamespace='_ligand.mol2', namespace='../../Data/temp/test/test',
                         altProNamespace='_protein.mol2',
                         altLigNamespace='_ligand.pdb')

# If there are errors and the code interrupts, you can set the startpoint to the last intermediate step
# If everything works you can combine all intermediate files to one big dataset
dataTrain = []
for f in os.scandir('../../Data/temp/train'):
    dataTrain.append(f.name)
dataTrain.sort()

trainLabels = Preprocessing.getLabels('../../Data/train/', '../../Data/index/INDEX_refined_data.2016')

newFile = h5py.File('../../Data/train.hdf5')
k = 0
for i in range(int(np.ceil(3767 / 50))):
    file = h5py.File('../../Data/temp/train/' + dataTrain[i], 'r')
    print('../../Data/temp/train/' + dataTrain[i])
    for j in range(50):
        try:
            newFile[str(k) + '/data'] = file[format(j)].value
            newFile[str(k) + '/label'] = trainLabels[k]
            k += 1
            print(k)
        except:
            continue

dataTest = []
for f in os.scandir('../../Data/temp/test'):
    dataTest.append(f.name)
dataTest.sort()

testsLabels = Preprocessing.getLabels('../../Data/test/', '../../Data/index/INDEX_refined_data.2016')

newFile = h5py.File('../../Data/test.hdf5')
k = 0
for i in range(int(np.ceil(290 / 50))):
    file = h5py.File('../../Data/temp/test/' + dataTest[i])
    for j in range(50):
        try:
            newFile[str(k) + '/data'] = file[format(j)].value
            newFile[str(k) + '/label'] = trainLabels[k]
            k += 1
            print(k)
        except:
            continue

# Now you should have two hdf5 files with the whole voxelised data and the labels of train and test data
# similar to the KDeep paper, which is mentioned above
