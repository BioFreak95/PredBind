# Import Area
import sys

sys.path.append('/home/max/Dokumente/Masterarbeit/PredBind')

import torch
import matplotlib.pyplot as plt
import numpy as np
import h5py
import schnetpack
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import pandas
import sys
from src.SchNet.tools.PreprocessingSchnet import PreprocessingSchnet
from src.SchNet.network.Shiftedsigmoid import ShiftedSigmoid

import torch.nn.functional as F
import torch
from torch.optim import Adam
from torch.autograd import Variable


class SchnetTraining:
    def train(self, resultfolder, traindb='Data/dataset_10_12_train_combined.db', benchdb='Data/dataset_10_12_test.db',
              traindata='../../Data/combined1618/', benchdata='../../Data/test/',
              indexpath='../../Data/INDEX_refined_data.2016.2018',
              properties=['KD'], threshold=10, cutoff=8, numVal=150, featureset=False,
              trainBatchsize=8, testBatchsize=1, natoms=None, props=False, ntrain=4444, ntest=290, use_noise=False,
              noise_mean=0.0, noise_std=0.1, chargeEmbedding=True,
              ownFeatures=False, nFeatures=8, finalFeature=None,
              max_z=200, n_atom_basis=20, n_filters=32,
              n_gaussians=25,
              normalize_filter=False, coupled_interactions=False,
              trainable_gaussians=False,
              n_interactions=5, distanceCutoff=2.5,
              cutoff_network=schnetpack.nn.cutoff.CosineCutoff, outputIn=32, outAggregation='avg', outLayer=2,
              outMode='postaggregate', outAct=schnetpack.nn.activations.shifted_softplus, outOutAct=None, n_acc_steps=8,
              remember=10,
              ensembleModel=False, n_epochs=150, lr=1e-3, weight_decay=0):

        print('Device: ', torch.cuda.current_device())
        torch.cuda.empty_cache()

        # Define Folder for Results
        Resultfolder = resultfolder
        act = ShiftedSigmoid()

        train_loader, val_loader, train, bench = self.createDataloader(traindb=traindb,
                                                                       benchdb=benchdb,
                                                                       traindata=traindata,
                                                                       benchdata=benchdata,
                                                                       indexpath=indexpath,
                                                                       properties=properties, threshold=threshold,
                                                                       cutoff=cutoff,
                                                                       numVal=numVal,
                                                                       featureset=featureset,
                                                                       trainBatchsize=trainBatchsize,
                                                                       testBatchsize=testBatchsize,
                                                                       natoms=natoms, props=props,
                                                                       ntrain=ntrain, ntest=ntest)

        model = schnetpack.representation.SchNet(use_noise=use_noise, noise_mean=noise_mean, noise_std=noise_std,
                                                 chargeEmbedding=chargeEmbedding,
                                                 ownFeatures=ownFeatures, nFeatures=nFeatures,
                                                 finalFeature=finalFeature,
                                                 max_z=max_z, n_atom_basis=n_atom_basis, n_filters=n_filters,
                                                 n_gaussians=n_gaussians,
                                                 normalize_filter=normalize_filter,
                                                 coupled_interactions=coupled_interactions,
                                                 trainable_gaussians=trainable_gaussians,
                                                 n_interactions=n_interactions, cutoff=distanceCutoff,
                                                 cutoff_network=cutoff_network)
        d = schnetpack.atomistic.Atomwise(n_in=outputIn, aggregation_mode=outAggregation,
                                          n_layers=outLayer,
                                          mode=outMode, activation=outAct, output_activation=outOutAct)

        model = schnetpack.AtomisticModel(model, d)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        print(model)
        print('number of trainable parameters =', SchnetTraining.count_parameters(model))

        # Defines Metrics, Hooks and Loss
        loss = SchnetTraining.mse_loss()
        # MSE Metrics for Validation
        metrics = [schnetpack.metrics.MeanSquaredError('KD', model_output='y')]
        # CSVHook -> Creates a log-file
        # ReduceLROnPlateauHook -> LR-Decay
        hooks = [schnetpack.train.CSVHook(log_path=Resultfolder, metrics=metrics),
                 schnetpack.train.ReduceLROnPlateauHook(optimizer, patience=7, factor=0.5, min_lr=1e-5),
                 schnetpack.train.EarlyStoppingHook(patience=40,
                                                    threshold_ratio=0.001)]

        # Create Trainer -> n_acc_steps is for accumulating gradient, if the graphiccard can not handle big batch-sizes
        trainer = schnetpack.train.Trainer(model_path=Resultfolder, model=model, loss_fn=loss,
                                           train_loader=train_loader,
                                           optimizer=optimizer, validation_loader=val_loader, hooks=hooks,
                                           n_acc_steps=n_acc_steps,
                                           keep_n_checkpoints=10, checkpoint_interval=5, remember=remember,
                                           ensembleModel=ensembleModel)

        # Start training on cuda for infinite epochs -> Use early stopping
        trainer.train(device='cuda', n_epochs=n_epochs)

    def createDataloader(self, traindb='Data/dataset_10_12_train_combined.db', benchdb='Data/dataset_10_12_test.db',
                         traindata='../../Data/combined1618/', benchdata='../../Data/test/',
                         indexpath='../../Data/INDEX_refined_data.2016.2018',
                         properties=['KD'], threshold=10, cutoff=8, numVal=150, featureset=False,
                         trainBatchsize=8, testBatchsize=1, natoms=None, props=False, ntrain=4444, ntest=290):

        train = schnetpack.data.AtomsData(traindb,
                                          available_properties=properties,
                                          environment_provider=schnetpack.environment.TorchEnvironmentProvider(cutoff,
                                                                                                               torch.device(
                                                                                                                   'cpu')))

        if featureset:
            if len(train) == 0:
                PreprocessingSchnet.createDatabaseFromFeatureset(train,
                                                                 threshold=threshold,
                                                                 featureFile=traindata, length=ntrain)
        else:
            if (len(train) == 0):
                PreprocessingSchnet.createDatabase(train, threshold=threshold, data_path=traindata,
                                                   index_path=indexpath)

        bench = schnetpack.data.AtomsData(benchdb,
                                          available_properties=properties,
                                          environment_provider=schnetpack.environment.TorchEnvironmentProvider(cutoff,
                                                                                                               torch.device(
                                                                                                                   'cpu')))

        if featureset:
            if len(bench) == 0:
                PreprocessingSchnet.createDatabaseFromFeatureset(bench,
                                                                 threshold=threshold,
                                                                 featureFile=benchdata,
                                                                 length=ntest)
        else:
            if (len(bench) == 0):
                PreprocessingSchnet.createDatabase(bench, data_path=benchdata, threshold=threshold,
                                                   index_path=indexpath)

        train, val, test = schnetpack.train_test_split(data=train, num_val=numVal, num_train=len(train) - numVal)

        print(len(train), len(bench), len(val))

        # Create Dataloader for Training
        train_loader = schnetpack.AtomsLoader(train, batch_size=trainBatchsize, shuffle=True, natoms=natoms,
                                              props=props)
        val_loader = schnetpack.AtomsLoader(val, batch_size=testBatchsize, shuffle=False, natoms=natoms, props=props)

        return train_loader, val_loader, train, bench

    @staticmethod
    def count_parameters(model):
        total_param = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                num_param = np.prod(param.size())
                if param.dim() > 1:
                    print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
                else:
                    print(name, ':', num_param)
                total_param += num_param
        return total_param

    @staticmethod
    def mse_loss():
        def loss_fn(batch, result):
            # N = Variable(torch.normal(torch.zeros(32),torch.ones(32)*0.1).cuda())
            diff = (batch['KD']) - result['y']
            err = torch.mean(diff ** 2)
            return err

        return loss_fn

    def plotting(self, project, traindb='Data/dataset_10_12_train_combined.db', benchdb='Data/dataset_10_12_test.db',
                 traindata='../../Data/combined1618/', benchdata='../../Data/test/',
                 indexpath='../../Data/INDEX_refined_data.2016.2018',
                 properties=['KD'], threshold=10, cutoff=8, numVal=150, featureset=False,
                 trainBatchsize=8, testBatchsize=1, natoms=None, props=False, ntrain=4444, ntest=290):

        train_loader, val_loader, train, bench = self.createDataloader(traindb=traindb,
                                                                       benchdb=benchdb,
                                                                       traindata=traindata,
                                                                       benchdata=benchdata,
                                                                       indexpath=indexpath,
                                                                       properties=properties,
                                                                       threshold=threshold,
                                                                       cutoff=cutoff,
                                                                       numVal=numVal,
                                                                       featureset=featureset,
                                                                       trainBatchsize=trainBatchsize,
                                                                       testBatchsize=testBatchsize,
                                                                       natoms=natoms, props=props,
                                                                       ntrain=ntrain, ntest=ntest)

        length1 = len(train)
        length2 = len(bench)

        torch.nn.Module.dump_patches = True
        best_model = torch.load(project + '/best_model')
        preds = []
        targets = []
        for count, batch in enumerate(val_loader):
            # move batch to GPU, if necessary
            batch = {k: v.to('cuda') for k, v in batch.items()}
            # apply model
            pred = best_model(batch)
            targets.append(batch['KD'].detach().cpu().numpy())
            preds.append(pred['y'].detach().cpu().numpy())

        targets_new = []
        preds_new = []
        for i in range(len(targets)):
            tar = targets[i]
            pre = preds[i]
            for j in range(len(tar)):
                targets_new.append(tar[j])
                preds_new.append(pre[j])

        preds_new = np.reshape(preds_new, length2)
        targets_new = np.reshape(targets_new, length2)

        pear = pearsonr(targets_new, preds_new)
        spear = spearmanr(targets_new, preds_new)

        diffs = []
        num_small = 0
        num_big = 0
        for i in range(length2):
            diffs.append((targets_new[i] - preds_new[i]) ** 2)
            if diffs[-1] < 1:
                num_small += 1
            if diffs[-1] > 5:
                num_big += 1

        print(np.max(diffs), np.min(diffs), np.var(diffs), np.mean(diffs), num_big, num_small)

        txt = ("MSE: " + str(np.round(np.mean(diffs), 4)) +
               ", RMSE: " + str(np.round(np.sqrt(np.mean(diffs)), 4)) +
               ", Pearson: " + str(np.round(pear[0], 4)) +
               ", Spearman: " + str(np.round(spear[0], 4)))

        x = targets_new
        y = preds_new

        fit = np.polyfit(x, y, 1)
        fit_fn = np.poly1d(fit)

        print(fit)

        # fit_fn is now a function which takes in x and returns an estimate for y

        plt.plot(x, y, 'yo', x, fit_fn(x), '--k')

        plt.xlim(0, 13)
        plt.ylim(2.5, 13)
        plt.xlabel('Target')
        plt.ylabel('Output of Network')
        # plt.title('Output of Pytorch-trained Network for Benchmark-Set')

        plt.text(0, 0, txt)
        plt.savefig(project + '/output.png', bbox_inches='tight')
        plt.clf()

        loss = pandas.read_csv(project + '/log.csv')['Train loss'].to_numpy()
        val_loss = pandas.read_csv(project + '/log.csv')['Validation loss'].to_numpy()
        lr = pandas.read_csv(project + '/log.csv')['Learning rate'].to_numpy()

        plt.plot(val_loss, label="Validation Loss")
        plt.plot(loss, label="Training Loss")
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        # plt.title('Train History')
        plt.legend()
        # plt.xlim(20,50)
        plt.ylim(0, 6)
        plt.savefig(project + '/training.png', bbox_inches='tight')
        plt.clf()

        plt.plot(lr)
        plt.xlabel('Epoch')
        plt.ylabel('lr')
        # plt.title('Learning Rate')
        plt.savefig(project + '/lr.png', bbox_inches='tight')
        plt.clf()

        targets = []
        preds = []
        for count, batch in enumerate(train_loader):
            # move batch to GPU, if necessary
            batch = {k: v.to('cuda') for k, v in batch.items()}
            # apply model
            pred = best_model(batch)
            targets.append(batch['KD'].detach().cpu().numpy())
            preds.append(pred['y'].detach().cpu().numpy())

        targets_new = []
        preds_new = []
        for i in range(len(targets)):
            tar = targets[i]
            pre = preds[i]
            for j in range(len(tar)):
                targets_new.append(tar[j])
                preds_new.append(pre[j])

        preds_new = np.reshape(preds_new, length1)
        targets_new = np.reshape(targets_new, length1)

        pear = pearsonr(targets_new, preds_new)
        spear = spearmanr(targets_new, preds_new)

        diffs = []
        num_small = 0
        num_big = 0

        for i in range(length1):
            diffs.append((targets_new[i] - preds_new[i]) ** 2)
            if diffs[-1] < 1:
                num_small += 1
            if diffs[-1] > 5:
                num_big += 1

        print(np.max(diffs), np.min(diffs), np.var(diffs), np.mean(diffs), num_big, num_small)

        txt = ("MSE: " + str(np.round(np.mean(diffs), 4)) +
               ", RMSE: " + str(np.round(np.sqrt(np.mean(diffs)), 4)) +
               ", Pearson: " + str(np.round(pear[0], 4)) +
               ", Spearman: " + str(np.round(spear[0], 4)))

        x = targets_new
        y = preds_new

        fit = np.polyfit(x, y, 1)
        fit_fn = np.poly1d(fit)

        print(fit)

        # fit_fn is now a function which takes in x and returns an estimate for y

        plt.plot(x, y, 'yo', x, fit_fn(x), '--k')

        plt.xlim(0, 13)
        plt.ylim(2.5, 13)
        plt.xlabel('Target')
        plt.ylabel('Output of Network')
        # plt.title('Output of Pytorch-trained Network for Training-Set')

        plt.text(0, 0, txt)
        plt.savefig(project + '/outputTrain.png', bbox_inches='tight')
        plt.clf()
