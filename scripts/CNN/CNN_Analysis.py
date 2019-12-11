# Here a Example for a Analysis will be added in future
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import h5py


# Output_plot

Path = 'Data/results/CNN/'

File = h5py.File(Path + 'Bench.hdf5', 'r')
labels = File['lables'].value
outs = File['outputs'].value

e2 = []
labels1 = []
outs1 = []
for i in range(len(labels)):
    e2.append((labels[i]-outs[i])**2)
    labels1.append(labels[i])
    outs1.append(outs[i])

txt = ("MSE: " + str(np.round(np.mean(e2), 4)) +
       ", RMSE: " + str(np.round(np.sqrt(np.mean(e2)), 4)) +
       ", Pearson: " + str(np.round(scipy.stats.pearsonr(labels1, outs1)[0], 4)) +
       ", Spearman: " + str(np.round(scipy.stats.spearmanr(labels1, outs1)[0], 4)))

x = labels1
y = outs1

fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit)
# fit_fn is now a function which takes in x and returns an estimate for y

plt.plot(x,y, 'yo', x, fit_fn(x), '--k')

plt.xlabel('Target')
plt.ylabel('Output of Network')
plt.title('Output of Pytorch-trained Network for Benchmark-Set')
plt.xlim(1,13)
plt.ylim(1,13)
plt.text(1, -1.5, txt)
plt.savefig(Path + 'Output_Pytorch_MSE_80_20.png', bbox_inches='tight')
File.close()
plt.clf()

# Hist-Plot

File = h5py.File(Path + 'history.hdf5', 'r')
test_mse = File['test_mse'][()]
train_mse = File['train_mse'][()]
test_rmse = File['test_rmse'][()]
train_rmse = File['train_rmse'][()]
plt.plot(test_mse, label="Validation Loss")
plt.plot(train_mse, label="Training Loss")
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.ylim(0,5)
plt.xlim(0,100)
plt.yticks(np.arange(0, 5, 0.5))
plt.legend()
plt.text(-1, -1, 'Lowest Val-loss: ' + str(np.min(test_mse)))
plt.axhline(np.min(test_mse), color='grey', linestyle=':')
plt.savefig(Path + 'TrainHist_Pytorch_MSE_80_20.png', bbox_inches='tight')
File.close()