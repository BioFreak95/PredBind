from src.CNN.networks.CNN import CNN
from src.CNN.networks.layers.OwnDataset import OwnDataset
from src.CNN.tools.Rotations import Rotations

import torch
from torch.utils.data import DataLoader

import numpy as np
import h5py


class Training:
    def __init__(self, model=None, optimizer=None):
        if model is None:
            self.model = CNN().cuda()
        else:
            self.model = model

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.99, 0.999), lr=1e-4)
        else:
            self.optimizer = optimizer

    def training(self, epoch, train_dataloader, dataset):
        self.model.train()
        mse_list = []
        rmse_list = []

        for batch_id, (data, target) in enumerate(train_dataloader):
            target = target.view(-1, 1)
            self.optimizer.zero_grad()
            data = data.float().cuda()
            target = target.float().cuda()
            out = self.model(data)
            criterion = torch.nn.MSELoss()
            loss = criterion(out, target)
            mse_list.append(loss.data.item())
            rmse_list.append(np.sqrt(loss.data.item()))
            loss.backward()
            self.optimizer.step()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} MSE-Mean: {:.6f} RMSE-Mean:  {:.6f}'.format(
                epoch, batch_id * len(data), dataset.__len__(),
                       100. * (batch_id * len(data)) / dataset.__len__(), loss.data.item(), np.mean(mse_list),
                np.mean(rmse_list)))

        print('Train Epoch: {} MSE (loss): {:.4f}, RMSE: {:.4f} Dataset length {}'.format(epoch, np.mean(mse_list),
                                                                                          np.mean(rmse_list),
                                                                                          dataset.__len__()))
        return np.mean(mse_list), np.mean(rmse_list)

    def testing(self, epoch, test_dataloader, dataset):
        self.model.eval()
        mse_list = []
        rmse_list = []

        for batch_id, (data, target) in enumerate(test_dataloader):
            target = target.view(-1, 1)
            data = data.float().cuda()
            target = target.float().cuda()
            out = self.model(data)
            criterion = torch.nn.MSELoss()
            loss = criterion(out, target)
            mse_list.append(loss.data.item())
            rmse_list.append(np.sqrt(loss.data.item()))

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} MSE-Mean: {:.6f} RMSE-Mean:  {:.6f}'.format(
                epoch, batch_id * len(data), dataset.__len__(),
                       100. * (batch_id * len(data)) / dataset.__len__(), loss.data.item(), np.mean(mse_list),
                np.mean(rmse_list)))

        print('Train Epoch: {} MSE (loss): {:.4f}, RMSE: {:.4f} Dataset length {}'.format(epoch, np.mean(mse_list),
                                                                                          np.mean(rmse_list),
                                                                                          dataset.__len__()))
        return np.mean(mse_list), np.mean(rmse_list)

    def benchmark(self, datapath='../Data/test.hdf5', nData = 290):
        rot = rotations()
        datafile = datapath
        labels = []
        outs = []
        for i in range(nData):
            outs1 = []
            target1 = []
            for j in range(24):
                with h5py.File(datafile, 'r') as file:
                    data = rot.rotation(data=file[str(i) + '/data'][()][0], k=j)
                    label = -np.log10(np.exp(-(file[str(i) + '/label'][()])))
                data = torch.from_numpy(data.reshape(1, 16, 24, 24, 24).copy()).float().cuda()
                out = self.model(data)
                outs1.append(out.cpu().data.numpy())
                target1.append(label)
            labels.append(np.mean(target1))
            outs.append(np.mean(outs1))
        error = []
        for i in range(290):
            error.append((outs[i]-labels[i])**2)
        print("testmean: ", np.mean(error), np.sqrt(np.mean(error)))

        return error, labels, outs

    def fit(self, epochs, train_path, result_datapath, kwargs=None, n_datapoints=3767, prct_train=0.8,
                 batch_size_train=128, batch_size_test=32):
        lowest_loss = np.inf
        train_mse = []
        test_mse = []
        train_rmse = []
        test_rmse = []
        if kwargs is None:
            kwargs = {'num_workers': 4}

        indices = np.arange(n_datapoints)
        train_size = int(prct_train * n_datapoints)
        test_size = n_datapoints - train_size
        train, test = torch.utils.data.random_split(indices, [train_size, test_size])

        train_set = OwnDataset(train, train_path)
        test_set = OwnDataset(test, train_path)
        train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size_train, shuffle=True, **kwargs)
        test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size_test, shuffle=False, **kwargs)

        for epoch in range(epochs):
            mse, rmse = self.training(epoch, train_dataloader, train_set)
            train_mse.append(mse)
            train_rmse.append(rmse)

            mse, rmse = self.testing(epoch, test_dataloader, test_set)
            test_mse.append(mse)
            test_rmse.append(rmse)

            if test_rmse[-1] < lowest_loss:
                lowest_loss = test_rmse[-1]
                torch.save(self.model.state_dict(), result_datapath + 'bestModel.pt')
            torch.save(self.model, result_datapath + 'lastModel.pt')
            torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, result_datapath + 'lastModel.tar')

        hist = h5py.File(result_datapath + 'history.hdf5', 'w')
        hist.create_dataset('train_mse', data=train_mse)
        hist.create_dataset('test_mse', data=test_mse)
        hist.create_dataset('train_rmse', data=train_rmse)
        hist.create_dataset('test_rmse', data=test_rmse)
        hist.close()
