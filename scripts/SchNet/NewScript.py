from src.SchNet.network.SchnetTraining import SchnetTraining
import schnetpack

nFilter = [64, 32, 16, 8]
nAtoms = [64, 32, 16, 8]
for a in nAtoms:
    for f in nFilter:
        schnet = SchnetTraining()
        resultfolder = 'resultsSchnet_' + str(a) + '_' + str(f)
        schnet.train(resultfolder=resultfolder, traindb='Data/dataset_80_10_train_combined_features.db',
                     benchdb='Data/dataset_80_10_test_features.db', traindata='Data/Schnet/trainSchnetKDeep.hdf5',
                     benchdata='Data/Schnet/testSchnetKDeep.hdf5',
                     indexpath='../../Data/INDEX_refined_data.2016.2018', properties=['KD', 'props'], threshold=10,
                     cutoff=8., numVal=150,
                     featureset=True, trainBatchsize=8, testBatchsize=1, natoms=None, props=False, ntrain=4444,
                     ntest=290,
                     use_noise=False, noise_mean=0.0, noise_std=0.1, chargeEmbedding=True, ownFeatures=False,
                     nFeatures=8,
                     finalFeature=None, max_z=100, n_atom_basis=a, n_filters=f, n_gaussians=100, normalize_filter=False,
                     coupled_interactions=False, trainable_gaussians=False, n_interactions=3, distanceCutoff=5.,
                     cutoff_network=schnetpack.nn.cutoff.CosineCutoff,
                     outputIn=a, outAggregation='avg', outLayer=2, outMode='postaggregate',
                     outAct=schnetpack.nn.activations.shifted_softplus, outOutAct=None,
                     n_acc_steps=8, remember=10, ensembleModel=False, n_epochs=150, lr=1e-3, weight_decay=0)

        schnet.plotting(project=resultfolder, traindb='Data/dataset_80_10_train_combined_features.db',
                        benchdb='Data/dataset_80_10_test_features.db', traindata='Data/Schnet/trainSchnetKDeep.hdf5',
                        benchdata='Data/Schnet/testSchnetKDeep.hdf5',
                        indexpath='../../Data/INDEX_refined_data.2016.2018', properties=['KD', 'props'], threshold=10,
                        cutoff=8., numVal=150,
                        featureset=True, trainBatchsize=8, testBatchsize=1, natoms=None, props=False, ntrain=4444,
                        ntest=290)
