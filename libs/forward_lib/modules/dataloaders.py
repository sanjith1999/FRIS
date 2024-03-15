import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import torch.nn.functional as F
from modules.datasets import *
import matplotlib.pyplot as plt


def get_tinyimagenet_dataloaders(img_size, train_batch_size ,torch_seed=10, task_type= 'phase2intensity', shrinkFactor = 1, **kwargs):
    '''
        Function to return train, validation TinyImageNet dataloaders
        Args:
            img_size         : Image size to resize
            train_batch_size : batch size for training
            torch_seed       : seed
        Returns:
            train_loader : Dataloader for training
            val_loader   : Dataloader for validation
    '''
    
    torch.manual_seed(torch_seed)
    
    data_dir= '/n/holyscratch01/wadduwage_lab/tinyimagenet/tiny-imagenet-200'

    my_transform= transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(), 
                    transforms.Resize((int(img_size//shrinkFactor), int(img_size//shrinkFactor))), 
                    transforms.CenterCrop((img_size, img_size))
                ])
    
    train_data = tinyimagenet_dataset(data_dir=data_dir, type_='train', transform=my_transform)
    val_data = tinyimagenet_dataset(data_dir=data_dir, type_='val', transform=my_transform)
    test_data = tinyimagenet_dataset(data_dir=data_dir, type_='test', transform=my_transform)
    
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, drop_last= True)
    val_loader = DataLoader(val_data, batch_size=train_batch_size, shuffle=False, drop_last= True)
    test_loader = DataLoader(test_data, batch_size=train_batch_size, shuffle=False, drop_last= True)

    return train_loader, val_loader, test_loader


def get_mnist_dataloaders(img_size, train_batch_size ,torch_seed=10, task_type= 'phase2amp', shrinkFactor = 1, biasOnoise=0, **kwargs):
    '''
        Function to return train, validation MNIST dataloaders
        Args:
            img_size         : Image size to resize
            train_batch_size : batch size for training
            torch_seed       : seed
        Returns:
            train_loader : Data loader for training
            val_loader   : Data loader for validation
    '''
    
    torch.manual_seed(torch_seed)
    
    data_dir= '/n/holyscratch01/wadduwage_lab/fypteam_22/datasets/temp_mnist'

    train_data = datasets.MNIST(root=data_dir, train=True, download=True)
    test_data = datasets.MNIST(root=data_dir, train=False, download=True)

    my_transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((int(img_size//shrinkFactor), int(img_size//shrinkFactor))), transforms.CenterCrop((img_size, img_size))])

    train_loader = DataLoader(mnist_dataset(data= train_data.data[:54000], labels= train_data.targets[:54000], transform= my_transform, task_type= task_type, biasOnoise=biasOnoise), batch_size=train_batch_size, shuffle=True, drop_last= True)
    val_loader = DataLoader(mnist_dataset(data= train_data.data[59000:], labels= train_data.targets[59000:], transform= my_transform, task_type= task_type, biasOnoise=biasOnoise), batch_size=32, shuffle=False, drop_last= True)
    test_loader = DataLoader(mnist_dataset(data= test_data.data, labels= test_data.targets, transform= my_transform, task_type= task_type, biasOnoise=biasOnoise), batch_size=32, shuffle=False, drop_last= True)

    return train_loader, val_loader, test_loader

def get_mnist_filtered_dataloaders(img_size, train_batch_size ,torch_seed=10, task_type= 'phase2amp', shrinkFactor = 1, biasOnoise=0, **kwargs):
    '''
        Function to return train, validation filtered MNIST dataloaders
        Args:
            img_size         : Image size to resize
            train_batch_size : batch size for training
            torch_seed       : seed
        Returns:
            train_loader : Data loader for training
            val_loader   : Data loader for validation
    '''
    
    torch.manual_seed(torch_seed)
    
    data_dir= '/n/holyscratch01/wadduwage_lab/fypteam_22/datasets/temp_mnist'

    train_data = datasets.MNIST(root=data_dir, train=True, download=True)
    test_data = datasets.MNIST(root=data_dir, train=False, download=True)

    my_transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((int(img_size//shrinkFactor), int(img_size//shrinkFactor))), transforms.CenterCrop((img_size, img_size))])

    train_loader = DataLoader(mnist_filtered_dataset(data= train_data.data[:54000], labels= train_data.targets[:54000], transform= my_transform, task_type= task_type, biasOnoise=biasOnoise), batch_size=train_batch_size, shuffle=True, drop_last= True)
    val_loader = DataLoader(mnist_filtered_dataset(data= train_data.data[59000:], labels= train_data.targets[59000:], transform= my_transform, task_type= task_type, biasOnoise=biasOnoise), batch_size=32, shuffle=False, drop_last= True)
    test_loader = DataLoader(mnist_filtered_dataset(data= test_data.data, labels= test_data.targets, transform= my_transform, task_type= task_type, biasOnoise=biasOnoise), batch_size=32, shuffle=False, drop_last= True)

    return train_loader, val_loader, test_loader

def get_mnist_dataloaders_rotate(img_size, train_batch_size ,torch_seed=10, task_type= 'phase2amp', shrinkFactor = 1, biasOnoise=0, **kwargs):
    '''
        Function to return train, validation MNIST dataloaders
        Args:
            img_size         : Image size to resize
            train_batch_size : batch size for training
            torch_seed       : seed
        Returns:
            train_loader : Data loader for training
            val_loader   : Data loader for validation
    '''
    angle = kwargs['cfg']['rotate_angle']
    torch.manual_seed(torch_seed)
    
    data_dir= '/n/holyscratch01/wadduwage_lab/fypteam_22/datasets/temp_mnist'

    train_data = datasets.MNIST(root=data_dir, train=True, download=True)
    test_data = datasets.MNIST(root=data_dir, train=False, download=True)

    my_transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((int(img_size//shrinkFactor), int(img_size//shrinkFactor))), transforms.CenterCrop((img_size, img_size)), transforms.functional.rotate(angle)])

    train_loader = DataLoader(mnist_dataset(data= train_data.data[:54000], labels= train_data.targets[:54000], transform= my_transform, task_type= task_type, biasOnoise=biasOnoise), batch_size=train_batch_size, shuffle=True, drop_last= True)
    val_loader = DataLoader(mnist_dataset(data= train_data.data[54000:], labels= train_data.targets[54000:], transform= my_transform, task_type= task_type, biasOnoise=biasOnoise), batch_size=32, shuffle=False, drop_last= True)

    return train_loader, val_loader

def get_mnist_dataloaders_smaller(img_size, train_batch_size ,torch_seed=10, task_type= 'phase2amp', shrinkFactor = 1, biasOnoise=0, **kwargs):
    '''
        Function to return train, validation MNIST dataloaders
        Args:
            img_size         : Image size to resize
            train_batch_size : batch size for training
            torch_seed       : seed
        Returns:
            train_loader : Data loader for training
            val_loader   : Data loader for validation
    '''
    
    torch.manual_seed(torch_seed)
    
    data_dir= '/n/holyscratch01/wadduwage_lab/fypteam_22/datasets/temp_mnist'

    train_data = datasets.MNIST(root=data_dir, train=True, download=True)
    test_data = datasets.MNIST(root=data_dir, train=False, download=True)

    my_transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((int(img_size//shrinkFactor), int(img_size//shrinkFactor))), transforms.CenterCrop((img_size, img_size))])

    train_loader = DataLoader(mnist_dataset(data= train_data.data[:54000], labels= train_data.targets[:54000], transform= my_transform, task_type= task_type, biasOnoise=biasOnoise), batch_size=train_batch_size, shuffle=True, drop_last= True)
    val_loader = DataLoader(mnist_dataset(data= train_data.data[54000:], labels= train_data.targets[54000:], transform= my_transform, task_type= task_type, biasOnoise=biasOnoise), batch_size=32, shuffle=False, drop_last= True)

    return train_loader, val_loader


def get_qpm_dataloaders(img_size, train_batch_size ,torch_seed=10, data_dir= '/n/holyscratch01/wadduwage_lab/D2NN_QPM_microscopy/datasets/qpm', task_type= 'phase2amp',shrinkFactor = 1, biasOnoise=0, **kwargs):
    '''
        Function to return train, validation QPM dataloaders
        Args:
            img_size         : Image size to resize
            train_batch_size : batch size for training
            torch_seed       : seed
            data_dir         : data directory which has the data hierachy as `./train/amp/00001.png`
        Returns:
            train_loader : Data loader for training
            val_loader   : Data loader for validation
    '''
    
    torch.manual_seed(torch_seed)
    
    my_transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((int(img_size//shrinkFactor), int(img_size//shrinkFactor))), transforms.CenterCrop((img_size, img_size))])

    train_data= qpm_dataset(data_dir=data_dir, type_= 'train', transform= my_transform, task_type= task_type,biasOnoise=biasOnoise)
    val_data= qpm_dataset(data_dir=data_dir, type_= 'val', transform= my_transform, task_type= task_type,biasOnoise=biasOnoise)
    
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, drop_last= True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, drop_last= True)

    return train_loader, val_loader


def get_qpm_np_dataloaders(img_size, train_batch_size ,torch_seed=10, data_dir= '/n/holyscratch01/wadduwage_lab/D2NN_QPM_microscopy/datasets/qpm_np_v3', task_type= 'phase2amp',shrinkFactor = 1, biasOnoise=0, photon_count=1, cfg= None, **kwargs):
    '''
        Function to return train, validation QPM dataloaders
        Args:
            img_size         : Image size to resize
            train_batch_size : batch size for training
            torch_seed       : seed
            data_dir         : data directory which has the data hierachy as `./train/amp/00001.png`
        Returns:
            train_loader : Data loader for training
            val_loader   : Data loader for validation
    '''
    
    torch.manual_seed(torch_seed)
    # transforms.ToPILImage(), 
    my_transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((int(img_size//shrinkFactor), int(img_size//shrinkFactor))), transforms.CenterCrop((img_size, img_size))])

    train_data = qpm_np_dataset(data_dir=data_dir, type_= 'train', transform = my_transform, task_type= task_type, biasOnoise = biasOnoise, photon_count = photon_count, cfg= cfg)
    val_data   = qpm_np_dataset(data_dir=data_dir, type_= 'val',   transform = my_transform, task_type= task_type, biasOnoise = biasOnoise, photon_count = photon_count, cfg= cfg)
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, drop_last= True)
    val_loader = DataLoader(val_data, batch_size=15, shuffle=False, drop_last= True)

    return train_loader, val_loader


def get_aug_qpm_dataloaders(img_size, train_batch_size ,torch_seed=10, data_dir= '/n/holyscratch01/wadduwage_lab/D2NN_QPM_microscopy/datasets/qpm', task_type= 'phase2amp',biasOnoise=0, **kwargs):
    '''
        Function to return augmented train and validation QPM dataloaders
        Args:
            img_size         : Image size to resize
            train_batch_size : batch size for training
            torch_seed       : seed
            data_dir         : data directory which has the data hierachy as `./train/amp/00001.png`
        Returns:
            train_loader : Data loader for training
            val_loader   : Data loader for validation
    '''
    
    torch.manual_seed(torch_seed)
    
    my_transform= transforms.Compose([transforms.ToTensor(), 
                                      transforms.Resize((img_size, img_size)),
                                     ])

    train_data= qpm_dataset(data_dir=data_dir, type_= 'train', transform= my_transform,biasOnoise=biasOnoise)
    val_data= qpm_dataset(data_dir=data_dir, type_= 'val', transform= my_transform,biasOnoise=biasOnoise)
    
    horizontalFlip_transform = transforms.Compose([transforms.ToTensor(), 
                                                   transforms.Resize((img_size, img_size)),
                                                   transforms.RandomHorizontalFlip(p=1.0)
                                                 ])
    verticalFlip_transform = transforms.Compose([transforms.ToTensor(), 
                                                 transforms.Resize((img_size, img_size)),
                                                 transforms.RandomVerticalFlip(p=1.0)
                                               ])
    train_horizontalF = qpm_dataset(data_dir=data_dir, type_= 'train', transform= horizontalFlip_transform,biasOnoise=biasOnoise)
    train_verticalF = qpm_dataset(data_dir=data_dir, type_= 'train', transform= verticalFlip_transform,biasOnoise=biasOnoise)
    
    augmented_train = torch.utils.data.ConcatDataset([train_data, train_horizontalF, train_verticalF])
    
#     return augmented_train, val_data
    
    train_loader = DataLoader(augmented_train, batch_size=train_batch_size, shuffle=True, drop_last= True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, drop_last= True)

    return train_loader, val_loader


def get_patchmnist_dataloaders(img_size, train_batch_size ,torch_seed=10, data_dir= '/n/holyscratch01/wadduwage_lab/D2NN_QPM_microscopy/datasets/mnistgrid_imgsize(32)', task_type= 'phase2amp',biasOnoise=0, **kwargs):  
    trainset = patchmnist_dataset(img_size, 'train', data_dir, 9500, task_type= task_type,biasOnoise=biasOnoise)
    valset= patchmnist_dataset(img_size, 'val', data_dir, 16, task_type= task_type,biasOnoise=biasOnoise)
    testset= patchmnist_dataset(img_size, 'test', data_dir, 16, task_type= task_type,biasOnoise=biasOnoise)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, drop_last= True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=16, shuffle=False, drop_last= False) # batch_sizes fixed
    test_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, drop_last= False) # batch_sizes fixed

    plt.figure()
    x, y= next(iter(val_loader))
    plt.imshow(x[0,0].angle())
    plt.title('Phase of sample datapoint : (from val loader)')
    plt.show()
        
    print('dataset magnitude range : ',x.abs().min().item(), x.abs().max().item())
    print('dataset phase range : ',x.angle().min().item(), x.angle().max().item())
    
    return train_loader, val_loader


def get_wide_dataloaders(img_size, train_batch_size ,torch_seed=10, data_dir= '/n/holyscratch01/wadduwage_lab/D2NN_mixed_noise/wide_dataset', task_type= 'phase2amp', shrinkFactor = 1,biasOnoise=0, **kwargs):
    '''
        Function to return train, validation "Noise" dataloaders
        Args:
            img_size         : Image size to resize
            train_batch_size : batch size for training
            torch_seed       : seed
            data_dir         : data directory which has the data hierarchy as `./train/P1024_0.npy`
        Returns:
            train_loader : Data loader for training
            val_loader   : Data loader for validation
    '''
    
    torch.manual_seed(torch_seed)
    
    my_transform= transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor(), transforms.Resize((int(img_size//shrinkFactor), int(img_size//shrinkFactor))), transforms.CenterCrop((img_size, img_size))])

    train_data = wide_dataset(data_dir=data_dir, type_= 'train', transform= my_transform, task_type= task_type)
    val_data   = wide_dataset(data_dir=data_dir, type_= 'val'  , transform= my_transform, task_type= task_type)
    
    train_loader = DataLoader(train_data, batch_size = train_batch_size, shuffle=True,  drop_last= True)
    val_loader   = DataLoader(val_data,   batch_size = 16              , shuffle=False, drop_last= True)

    return train_loader, val_loader

def get_bacteria_dataloaders(img_size, train_batch_size ,torch_seed=10, data_dir= '/n/holyscratch01/wadduwage_lab/D2NN_QPM_classification/datasets/bacteria_np', task_type= 'phase2amp',shrinkFactor = 1, biasOnoise=0, photon_count=1, cfg= None, **kwargs):
    '''
        Function to return train, validation QPM dataloaders
        Args:
            img_size         : Image size to resize
            train_batch_size : batch size for training
            torch_seed       : seed
            data_dir         : data directory which has the data hierachy as `./train/amp/00001.png`
        Returns:
            train_loader : Data loader for training
            val_loader   : Data loader for validation
    '''
    
    torch.manual_seed(torch_seed)
    # transforms.ToPILImage(), 
    my_transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((int(img_size//shrinkFactor), int(img_size//shrinkFactor))), transforms.CenterCrop((img_size, img_size))])

    train_data = bacteria_dataset(data_dir=data_dir, type_= 'train', transform = my_transform, task_type= task_type, biasOnoise = biasOnoise, photon_count = photon_count, cfg= cfg)
    val_data   = bacteria_dataset(data_dir=data_dir, type_= 'val',   transform = my_transform, task_type= task_type, biasOnoise = biasOnoise, photon_count = photon_count, cfg= cfg)
    
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, drop_last= True)
    val_loader = DataLoader(val_data, batch_size=15, shuffle=False, drop_last= True)

    return train_loader, val_loader


def get_rbc_dataloaders(img_size, train_batch_size ,torch_seed=10, data_dir= '/n/holyscratch01/wadduwage_lab/fypteam_22/datasets/rbc/processed/', task_type= 'phase2amp',shrinkFactor = 1, photon_count=1, cfg= None, **kwargs):
    '''
        Function to return train, validation red blood cells dataloaders
        Args:
            img_size         : Image size to resize
            train_batch_size : batch size for training
            torch_seed       : seed
            data_dir         : data directory which has the data hierachy as `./phase/Phase_H1_220126_1.npy`
        Returns:
            train_loader : Data loader for training
            val_loader   : Data loader for validation
    '''
    
    torch.manual_seed(torch_seed)
    # transforms.ToPILImage(), 
    my_transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((int(img_size//shrinkFactor), int(img_size//shrinkFactor))), transforms.CenterCrop((img_size, img_size))])

    train_data = rbc_dataset(data_dir=data_dir, type_= 'train', transform= my_transform, task_type= task_type, photon_count=photon_count, cfg= cfg)
    val_data   = rbc_dataset(data_dir=data_dir, type_= 'val', transform= my_transform, task_type= task_type, photon_count=photon_count, cfg= cfg)
    test_data  = rbc_dataset(data_dir=data_dir, type_= 'test', transform= my_transform, task_type= task_type, photon_count=photon_count, cfg= cfg)
    
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, drop_last= False)
    val_loader = DataLoader(val_data, batch_size=15, shuffle=False, drop_last= True)
    test_loader = DataLoader(test_data, batch_size=15, shuffle=False, drop_last= True)

    return train_loader, val_loader, test_loader