import os
from os import path
import pickle
# import logging
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset , Dataset
from torchvision.datasets import CIFAR10, MNIST #, CIFAR100, STL10, ImageFolder
import torchvision.transforms as transforms
import myConfig

CGEN = '\033[92m'
CEND = '\033[0m'

class LoadData:
    def __init__(self):
        pass

    @staticmethod
    def load_image(dataset_name):
        load_data = LoadData()
        if dataset_name == 'mnist':
            return load_data.load_mnist()
            # pass
        elif dataset_name == 'cifar10':
            return load_data.load_cifar10()


    def load_cifar10_data(self):
        train_loader, test_loader, train_set, test_set, classes = LoadData.load_cifar10()
        print(CGEN + 'load_cifar10 PASS' + CEND)
        return train_set
    
    @staticmethod
    def load_cifar10(batch_size=32, num_workers=1):
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = CIFAR10(root=myConfig.DATASET_PATH + 'Cifar10', train=True, download=True, transform=transform_train)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_set = CIFAR10(root=myConfig.DATASET_PATH + 'Cifar10', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, test_loader, train_set, test_set, classes  

    @staticmethod
    def load_mnist(batch_size=32,num_workers=1):
        classes = ( '0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        """
        transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = MNIST(root=myConfig.DATASET_PATH + 'mnist', train=True, transform=transform,
                          download=True)
        test_set = MNIST(root=myConfig.DATASET_PATH + 'mnist', train=False, transform=transform)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,num_workers=num_workers)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False,num_workers=num_workers)
        return train_loader, test_loader, train_set, test_set, classes


class DataStore:
    def __init__(self):
        pass
    
    def create_basic_folders(self):
        myConfig.path_create()
 
        
    def create_folder(self, folder):
        if not path.exists(folder):
            try:
                
                print(f'checking directory {folder}')
                os.mkdir(folder)
                
                print(f'new directory {folder} created')
            except OSError as error:
                print(f'deleting old and creating new empty {folder}')
                shutil.rmtree(folder)
                os.mkdir(folder)
                print(f'new empty directory {folder} created')
        else:
            print(f'folder {folder} exists, do not need to create again.')

    def save_data(self, data_arg, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data_arg, f)

    def load_data(self,filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


class attacker_dataset(Dataset):
 
    def __init__(self,price_df, transform=None):
        # price_df=pd.read_csv(file_name)
        self.transform = transform
        y = price_df.loc[:, 'label'].values
        x = price_df.loc[:, 'data'].values
        a = np.array([], dtype = np.float32)
        for i in range(len(x)):
            for j in range(len(x[i])):
                a = np.append(a, np.array(x[i][j], dtype = np.float32))
        a = a.reshape(len(x),1, -1, 10)
        self.x_train = a
        self.y_train = y

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        if self.transform is not None:
            x_tra = self.transform(self.x_train[idx])
        else:
            x_tra = self.x_train[idx]
        return x_tra, self.y_train[idx]


def get_dataloader(df, feature_name = 'data', batch_size = 32, shuffle=False):
    y = df.loc[:, 'label'].values
    x = df.loc[:, feature_name].values

    x_len = len(x)
    
    a = np.array([], dtype = np.float32)
    for i in range(len(x)):
        for j in range(len(x[i])):
            a = np.append(a, np.array(x[i][j], dtype = np.float32))
    a = a.reshape(x_len, 1, -1, 10)
    
    x_tensor = torch.from_numpy(a)
    y_tensor = torch.from_numpy(y)
    print(f'x_tensor[0] :\n{x_tensor[0]}')
    data_set = TensorDataset(x_tensor,y_tensor)

    # transform = transforms.Compose([
            # transforms.Resize(32),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])

    dataloader = DataLoader(data_set, batch_size = batch_size, shuffle=shuffle, num_workers=1, drop_last = True)
    return dataloader