from src.CNN.networks.Training import Training
import torch
import h5py

print(torch.cuda.current_device())
training = Training()

# Start training
training.fit(epochs=100, train_path='Data/train.hdf5', result_datapath='Data/results/CNN/', n_datapoints=3767, ensemble=False,
             remember=10, augmentation=True, batch_size_train=32, prct_train=0.85, test_path='Data/test.hdf5')

# If you use the same paths as used in CNN_Preprocessing than you can use default values

# Start Benchmarl
error, labels, outputs = training.benchmark(datapath='Data/test.hdf5', n_datapoints=290, ensemble=False,
                                            model='Data/results/CNN/bestmodels/', rotations=True)

# Safe results
file = h5py.File('Data/results/CNN/Bench.hdf5')
file['squaredErrors'] = error
file['lables'] = labels
file['outputs'] = outputs
