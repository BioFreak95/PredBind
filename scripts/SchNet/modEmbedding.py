from src.SchNet.tools.PreprocessingSchnet import PreprocessingSchnet
from src.SchNet.network.Shiftedsigmoid import ShiftedSigmoid

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import schnetpack

import torch.nn.functional as F
import torch
from torch.optim import Adam
from torch.autograd import Variable

# Prepare Cuda -> Never ever use CPU -> Extremely slow
print('Device: ', torch.cuda.current_device())
torch.cuda.empty_cache()

# Define Folder for Results
Resultfolder = 'resultsSchnet'

# Create Datasets Benchmark and Training -> Split Training in Train and Val set

train = schnetpack.data.AtomsData('Data/dataset_10_12_train_combined_features.db', available_properties=['KD', 'props'],
                                  environment_provider=schnetpack.environment.TorchEnvironmentProvider(10.,
                                                                                                       torch.device(
                                                                                                           'cpu')))
if len(train) == 0:
    PreprocessingSchnet.createDatabaseFromFeatureset(train, threshold=12, featureFile='Data/Schnet/trainSchnetKDeep',
                                                     length=4444)

bench = schnetpack.data.AtomsData('Data/dataset_10_12_test_features.db', available_properties=['KD', 'props'],
                                  environment_provider=schnetpack.environment.TorchEnvironmentProvider(10.,
                                                                                                       torch.device(
                                                                                                           'cpu')))
if len(bench) == 0:
    PreprocessingSchnet.createDatabaseFromFeatureset(bench, featureFile='Data/Schnet/testSchnetKDeep', threshold=12,
                                                     length=290)

train, val, test = schnetpack.train_test_split(data=train, num_val=150, num_train=len(train) - 150)

print(len(train), len(bench), len(val))

# Create Dataloader for Training
train_loader = schnetpack.AtomsLoader(train, batch_size=8, shuffle=True, natoms=None, props=False)
val_loader = schnetpack.AtomsLoader(val, batch_size=1, shuffle=False, natoms=None, props=False)

# Call ShiftedSigmoid for Activation -> You can use, but you do not have to
act = ShiftedSigmoid()

# Create Model and Optimizer -> Please Note, that here a modified Fork from the original schnetpack is used
# This allows a Noise on the Positions, also the reducing Featurevektor in interaction-layer is part of the modification
model = schnetpack.representation.SchNet(use_noise=False, noise_mean=0.0, noise_std=0.1, chargeEmbedding = True,
                                         ownFeatures = False, nFeatures = 8, finalFeature = None,
                                         max_z=200, n_atom_basis=20, n_filters=[32, 24, 16, 8, 4], n_gaussians=25,
                                         normalize_filter=False, coupled_interactions=False, trainable_gaussians=False,
                                         n_interactions=5, cutoff=2.5,
                                         cutoff_network=schnetpack.nn.cutoff.CosineCutoff)
# Modification of schnetpack allows an activation-function on the output of the output-network
d = schnetpack.atomistic.Atomwise(n_in=20, aggregation_mode='avg',
                                  n_layers=4, mode='postaggregate') # , activation=F.relu)#, output_activation=act.forward)
model = schnetpack.AtomisticModel(model, d)
optimizer = Adam(model.parameters())


# This function counts the number of trainable parameters of the network
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


# Print Model-Structure and number of trainable parameters
print(model)
print('number of trainable parameters =', count_parameters(model))


# Defines the Loss for schnetpack-Trainer -> Allows an Noise at the labels
def mse_loss():
    def loss_fn(batch, result):
        # N = Variable(torch.normal(torch.zeros(32),torch.ones(32)*0.1).cuda())
        diff = (batch['KD']) - result['y']
        err = torch.mean(diff ** 2)
        return err

    return loss_fn


# Defines Metrics, Hooks and Loss
loss = mse_loss()
# MSE Metrics for Validation
metrics = [schnetpack.metrics.MeanSquaredError('KD', model_output='y')]
# CSVHook -> Creates a log-file
# ReduceLROnPlateauHook -> LR-Decay
hooks = [schnetpack.train.CSVHook(log_path=Resultfolder, metrics=metrics),
         schnetpack.train.ReduceLROnPlateauHook(optimizer, patience=7, factor=0.5, min_lr=1e-5),
         schnetpack.train.EarlyStoppingHook(patience=40,
                                            threshold_ratio=0.001)]

# Create Trainer -> n_acc_steps is for accumulating gradient, if the graphiccard can not handle big batch-sizes
trainer = schnetpack.train.Trainer(model_path=Resultfolder, model=model, loss_fn=loss, train_loader=train_loader,
                                   optimizer=optimizer, validation_loader=val_loader, hooks=hooks, n_acc_steps=8,
                                   keep_n_checkpoints=10, checkpoint_interval=5, remember=10, ensembleModel=False)

# Start training on cuda for infinite epochs -> Use early stopping
trainer.train(device='cuda', n_epochs=150)

# Validation Area

# Open the best Model
best_model = torch.load(Resultfolder + '/best_model')
bench_loader = schnetpack.AtomsLoader(bench, batch_size=1)
preds = []
targets = []

# Make predictions with benchmark-set
for count, batch in enumerate(bench_loader):
    batch = {k: v.to('cuda') for k, v in batch.items()}
    pred = best_model(batch)
    targets.append(batch['KD'].detach().cpu().numpy())
    preds.append(pred['y'].detach().cpu().numpy())

# Reshape the Predictions and the Targets -> List which can not be transformed with numpy
targets_new = []
preds_new = []

for i in range(len(targets)):
    tar = targets[i]
    pre = preds[i]
    for j in range(len(tar)):
        targets_new.append(tar[j])
        preds_new.append(pre[j])

preds_new = np.reshape(preds_new, 290)
targets_new = np.reshape(targets_new, 290)

# Calculate Pearson and Spearman Correlation
pear = pearsonr(targets_new, preds_new)
spear = spearmanr(targets_new, preds_new)

# Calculate MSE -> Additionally count number of predictions with high errors and low errors
diffs = []
num_small = 0
num_big = 0
for i in range(290):
    diffs.append((targets_new[i] - preds_new[i]) ** 2)
    if diffs[-1] < 1:
        num_small += 1
    if diffs[-1] > 5:
        num_big += 1

# Create String for Log-File
string = 'spear: ' + str(spear[0]) + '\n' + 'pear: ' + str(pear[0]) + '\n' + 'max: ' + str(
    np.max(diffs)) + '\n' + 'min: ' + str(np.min(diffs)) + '\n' + 'std: ' + str(np.std(diffs)) + '\n' + 'var: ' + str(
    np.var(diffs)) + '\n' + 'mean: ' + str(np.mean(diffs)) + '\n' + 'n < 1: ' + str(num_small) + '\n' + 'n > 5: ' + str(
    num_big) + '\n'

# Write Analysis-String in Log-File
with open(Resultfolder + '/log.csv', 'a') as file:
    file.write(string)

# Write Diffs in Log-File
with open(Resultfolder + '/log.csv', 'a') as f:
    for item in diffs:
        f.write("%s\n" % item)
