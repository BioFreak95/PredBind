from src.CNN.networks.Training import Training
import torch
import h5py

print(torch.cuda.current_device())

training = Training()
training.fit(epochs=50, train_path='../Data/train.hdf5', result_datapath='../Data/results/', n_datapoints=3767)

# If you use the same paths as used in CNN_Preprocessing than you can use default values

error, labels, outputs = training.benchmark()

file = h5py.File('../Data/results/Bench.hdf5')
file['squaredErrors'] = error
file['lables'] = labels
file['outputs'] = outputs

# You can now use the Bench and Hist files to analyze the results
