import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import torch.nn.functional as F
import glob
import cv2
import os
import torchvision
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("default")

class tinyimagenet_dataset(torch.utils.data.Dataset):
    '''
        A standard dataset class to get TinyImageNet dataset
        
        Args:
            data_dir  : data directory which contains data hierarchy as `./train/class/image1.JPEG` or `.val/image1.JPEG`
            type_     : whether dataloader if train/ val 
            transform : torchvision.transforms
    '''
    def __init__(self, data_dir, type_='train', transform= None, task_type= 'phase2intensity', **kwargs):
        assert type_ in ['train', 'val', 'test']
        
        self.transform = transform
        self.task_type = task_type
        
        self.img_dirs = datasets.ImageFolder(os.path.join(data_dir, type_))

    def __len__(self):
        return len(self.img_dirs) 
        
    def __getitem__(self, index):
        phase_img, _ = self.img_dirs[index]
        phase_img = self.transform(phase_img)
        
        cmplx_img = torch.exp(1j*phase_img*np.pi)
        
        return cmplx_img, 0
    

class qpm_dataset(torch.utils.data.Dataset):
    '''
        A standard dataset class to get QPM dataset
        
        Args:
            data_dir  : data directory which contains data hierarchy as `./train/amp/00001.png`
            type_     : whether dataloader if train/ val 
            transform : torchvision.transforms
    '''
    
    def __init__(self, data_dir='datasets/qpm', type_= 'train', transform= None, task_type= 'phase2amp',biasOnoise=0, **kwargs):
        self.transform= transform
        self.task_type= task_type
        self.biasOnoise = biasOnoise

        self.amp_img_dirs = sorted(glob.glob(f'{data_dir}/{type_}/amp/*'), key= lambda x: int(x.split('/')[-1][:-4]))
        self.phase_img_dirs = sorted(glob.glob(f'{data_dir}/{type_}/phase/*'), key= lambda x: int(x.split('/')[-1][:-4]))
        
        assert len(self.amp_img_dirs) == len(self.phase_img_dirs), 'Number of phase and amp images are different !!!'
    def __len__(self):
        return len(self.amp_img_dirs)
        
    def __getitem__(self, idx): 
        amp_img = Image.fromarray(cv2.cvtColor(cv2.imread(self.amp_img_dirs[idx]), cv2.COLOR_RGBA2GRAY))
        phase_img = Image.fromarray(cv2.cvtColor(cv2.imread(self.phase_img_dirs[idx]), cv2.COLOR_RGBA2GRAY))

        amp_img= self.transform(amp_img)
        phase_img= self.transform(phase_img) + self.biasOnoise
        
        if self.task_type=='phase2amp' or self.task_type=='phase2intensity':
            qpm_img= amp_img * torch.exp(1j*phase_img*np.pi)
        elif self.task_type == 'phasenoamp2amp':
            qpm_img= torch.exp(1j*phase_img*np.pi)
        elif self.task_type=='amp2amp':
            qpm_img= phase_img
        elif self.task_type=='ampphase2amp':
            qpm_img= phase_img * torch.exp(1j*phase_img*np.pi)
        return qpm_img, 0

    
class qpm_np_dataset(torch.utils.data.Dataset):
    '''
        A standard dataset class to get QPM dataset
        
        Args:
            data_dir  : data directory which contains data hierarchy as `./train/amp/00001.png`
            type_     : whether dataloader if train/ val 
            transform : torchvision.transforms
    '''
    
    def __init__(self, data_dir='datasets/qpm_np_v3', type_= 'train', transform= None, task_type= 'phase2amp',biasOnoise=0, photon_count=1, cfg= None, **kwargs):
        self.transform= transform
        self.task_type= task_type
        self.biasOnoise = biasOnoise
        self.photon_count = photon_count
        self.cfg= cfg

        self.amp_img_dirs = sorted(glob.glob(f'{data_dir}/{type_}/amp/*'), key= lambda x: int(x.split('/')[-1][:-4]))
        self.phase_img_dirs = sorted(glob.glob(f'{data_dir}/{type_}/phase/*'), key= lambda x: int(x.split('/')[-1][:-4]))
        
        assert len(self.amp_img_dirs) == len(self.phase_img_dirs), 'Number of phase and amp images are different !!!'
    def __len__(self):
        return len(self.amp_img_dirs)
        
    def __getitem__(self, idx): 
        amp_img = np.load(self.amp_img_dirs[idx])
        phase_img = np.load(self.phase_img_dirs[idx])

        amp_img= self.transform(amp_img)
        phase_img= self.transform(phase_img) + self.biasOnoise
        
        if 'dataset_debug_opts' in self.cfg.keys():
            if 'clip_phase' in self.cfg['dataset_debug_opts']: # 'clip_phase@phase_set_pi'
                delta = 0.000001
                angle_max = eval(self.cfg['angle_max'])
                phase_img = torch.clip(phase_img, min = 0, max = (2*np.pi) - delta) #torch.clip(phase_img, min= -0.44, max= angle_max - delta - 0.44*2) + 0.44                
                
                if 'phase_set_pi' in self.cfg['dataset_debug_opts']:
                    phase_img = phase_img/(2*np.pi) * angle_max
                    text = "Warning : phase_img = phase_img/(2*np.pi)  * " + str(angle_max)
                    warnings.warn(text)
            else:
                raise ValueError(f"no 'clip_phase' in dataloader --> SSIM calculation will be wrong !!!")
        else:
            raise ValueError(f"no 'dataset_debug_opts: clip_phase' in dataloader --> SSIM calculation will be wrong !!!")
                
        if self.task_type=='phase2amp' or self.task_type=='phase2intensity':
            qpm_img = self.photon_count * amp_img * torch.exp(1j*phase_img)   
        else:
            raise NotImplementedError(f"Code should be verified for task_type : {task_type}")
        #elif self.task_type == 'phasenoamp2amp':
        #    qpm_img= torch.exp(1j*phase_img)
        #elif self.task_type=='amp2amp':
        #    qpm_img= phase_img
        #elif self.task_type=='ampphase2amp':
        #    qpm_img= phase_img * torch.exp(1j*phase_img)
        return qpm_img, phase_img
    

class mnist_dataset(torch.utils.data.Dataset):
    '''
        A standard dataset class to get mnist dataset
        
        Args:
            data      : Numpy arrays (n_samples, 1, image_size, image_size) ## to be completed
            labels    : Numpy array of targets (n_samples,) ## to be completed
            transform : torchvision.transforms
    '''
    
    def __init__(self, data= None, labels= None,transform= None,task_type= 'phase2amp', biasOnoise = 0, **kwargs):
        self.transform= transform
        self.task_type= task_type
        self.biasOnoise = biasOnoise

        self.data= np.array(data)
        self.labels= np.array(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        transformed_img = self.transform(Image.fromarray(self.data[idx])) + self.biasOnoise
        
        if self.task_type=='phase2amp' or self.task_type=='phase2intensity':
            mnist_img = torch.exp(1j*transformed_img*np.pi)  # Convert input ground truth images to phase images
        elif self.task_type=='amp2amp':
            mnist_img= transformed_img # add bias/ noise
        elif self.task_type=='ampphase2amp':
            mnist_img = transformed_img*torch.exp(1j*transformed_img*np.pi)
        
        return mnist_img, torch.tensor(self.labels[idx])


class mnist_filtered_dataset(torch.utils.data.Dataset):
    '''
        A standard dataset class to get filtered mnist dataset
        
        Args:
            data      : Numpy arrays (n_samples, 1, image_size, image_size) ## to be completed
            labels    : Numpy array of targets (n_samples,) ## to be completed
            transform : torchvision.transforms
    '''
    
    def __init__(self, data= None, labels= None,transform= None,task_type= 'phase2amp', biasOnoise = 0, **kwargs):
        self.transform= transform
        self.task_type= task_type
        self.biasOnoise = biasOnoise

        self.data= np.array(data)
        self.labels= np.array(labels)

        ### filter dataset to have digits only 5 or above
        mask = self.labels >= 5
        self.data   = self.data[mask]
        self.labels = self.labels[mask]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        transformed_img = self.transform(Image.fromarray(self.data[idx])) + self.biasOnoise
        
        if self.task_type=='phase2amp' or self.task_type=='phase2intensity':
            mnist_img = torch.exp(1j*transformed_img*np.pi)  # Convert input ground truth images to phase images
        elif self.task_type=='amp2amp':
            mnist_img= transformed_img # add bias/ noise
        elif self.task_type=='ampphase2amp':
            mnist_img = transformed_img*torch.exp(1j*transformed_img*np.pi)
        
        return mnist_img, torch.tensor(self.labels[idx])
    
    
class patchmnist_dataset(torch.utils.data.Dataset):
    def __init__(self, img_size= 32, type_= 'train', img_dir= None, num_samples= None,task_type= 'phase2amp',biasOnoise=0, **kwargs):
        super(patchmnist_dataset, self).__init__()
        
        self.type_ = type_
        img_list = sorted(glob.glob(f"{img_dir}/{self.type_}/*.jpg"), key= lambda x: int(x.split('/')[-1][:-4]))
        print(f'total images found in: {img_dir}/{self.type_} -> {len(img_list)}')
        
        if num_samples==None:num_samples=len(img_list)
            
        if len(img_list)<num_samples:
            print(f'WARNING -> Dataset: len(images) < num_samples -> num_samples will be neglected !!!')
            self.img_list= img_list
        else:
            self.img_list= img_list[:num_samples]
        
        
        self.mean =0
        self.std=1
        
        self.transform = transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((self.mean,), (self.std,)),
                                torchvision.transforms.RandomCrop((img_size,img_size), padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),
                                torchvision.transforms.Grayscale(num_output_channels=1)])
        self.task_type= task_type
        self.biasOnoise = biasOnoise
        
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        torch.manual_seed(idx)  # explanation: below
        
        #apply same random crop for image with same idx. What this do is, creating a unique random crop margins for each image in the dataset (can be train/ validation/ test). So it makes this random cropped datasets fixed as normal datasets but preserving the complexity.
        # Note that, if we apply this only to validation and test sets, it results unlimited large training set because for every epoch, it generates entirely new batches. This will not be similar to real situations where we have limited amount of data. So it is better to keep the randomness fixed for each image. 
        
        img = self.transform(Image.fromarray(plt.imread(self.img_list[idx]))) + self.biasOnoise
        
        if self.task_type=='phase2amp' or self.task_type=='phase2intensity':
            mnist_patch_img = torch.exp(1j*img*np.pi)  # Convert input ground truth images to phase images
        elif self.task_type=='amp2amp':
            mnist_patch_img = img
        elif self.task_type=='ampphase2amp':
            mnist_patch_img = img*torch.exp(1j*img*np.pi)
        
        torch.manual_seed(np.random.randint(0, 500000))
        return mnist_patch_img, torch.tensor(0) # 2nd output-> to maintain consistency across all MNIST dataloaders 

class wide_dataset(torch.utils.data.Dataset):
    '''
        A standard dataset class to get QPM dataset
        
        Args:
            data_dir  : data directory which contains data hierarchy as `./train/P1024_0.npy`
            type_     : whether dataloader if train/ val 
            transform : torchvision.transforms
    '''
    
    def __init__(self, data_dir='datasets/wide_dataset', type_= 'train', transform= None, task_type= 'phase2amp', **kwargs):
        self.transform= transform
        self.task_type= task_type
        self.complex_img_dirs   = sorted(glob.glob(f'{data_dir}/{type_}/*'))
       
    def __len__(self):
        return len(self.complex_img_dirs)

    def __getitem__(self, idx): 
        complex_img = np.load(self.complex_img_dirs[idx])

        amp_img_   = Image.fromarray(np.abs(complex_img).astype('float64'), 'RGB')
        phase_img_ = Image.fromarray(np.angle(complex_img).astype('float64'), 'RGB')

        amp_img   = self.transform(amp_img_)
        phase_img = self.transform(phase_img_)
        
        if self.task_type=='phase2amp' or self.task_type=='phase2intensity':
            qpm_img= amp_img * torch.exp(1j*phase_img*np.pi)
        elif self.task_type == 'phasenoamp2amp':
            qpm_img= torch.exp(1j*phase_img*np.pi)
        elif self.task_type=='amp2amp':
            qpm_img= phase_img
        elif self.task_type=='ampphase2amp':
            qpm_img= phase_img * torch.exp(1j*phase_img*np.pi)
        return qpm_img, 0

class bacteria_dataset(torch.utils.data.Dataset):
    '''
        A standard dataset class to get the bacteria dataset
        
        Args:
            data_dir  : data directory which contains data hierarchy as `./train/amp/00001.png`
            type_     : whether dataloader if train/ val 
            transform : torchvision.transforms
    '''
    
    def __init__(self, data_dir='datasets/bacteria_np', type_= 'train', transform= None, task_type= 'phase2amp',biasOnoise=0, photon_count=1, cfg= None, **kwargs):
        self.transform= transform
        self.task_type= task_type
        self.cfg = cfg
        
        self.biasOnoise = biasOnoise
        self.photon_count = photon_count

        self.phase_img_dirs = sorted(glob.glob(f'{data_dir}/{type_}/*/*'), key= lambda x: int(x.split('/')[-1][:-4]))
        self.phase_img_dirs = self.phase_img_dirs[:int(len(self.phase_img_dirs)*0.15)]
        
    def __len__(self):
        return len(self.phase_img_dirs)
        
    def __getitem__(self, idx): 
        phase_img = np.load(self.phase_img_dirs[idx], allow_pickle=True)[0].astype('float32')

        phase_img= self.transform(phase_img) + self.biasOnoise
        
        if 'dataset_debug_opts' in self.cfg.keys():
            if 'clip_phase' in self.cfg['dataset_debug_opts']: # 'clip_phase@phase_set_pi'
                delta = 0.000001
                angle_max = eval(self.cfg['angle_max'])
                phase_img = torch.clip(phase_img, min=0, max = (2*np.pi) - delta)
                #phase_img = torch.clip(phase_img, min=0, max = angle_max - delta) [We clip at different points, information changes => Not ideal!]
                

                if 'phase_set_pi' in self.cfg['dataset_debug_opts']:
                    phase_img = (phase_img/(2*np.pi))*angle_max
                    #phase_img = phase_img/angle_max * np.pi  [We clip at different points, information changes => Not ideal!]
                    
            else:
                raise ValueError(f"no 'clip_phase' in dataloader --> SSIM calculation will be wrong !!!")
        else:
            raise ValueError(f"no 'dataset_debug_opts: clip_phase' in dataloader --> SSIM calculation will be wrong !!!")
        
        if self.task_type=='phase2amp' or self.task_type=='phase2intensity' or self.task_type=='phasenoamp2intensity':
            qpm_img= torch.exp(1j*phase_img)   
        else:
            raise NotImplementedError(f"Code should be verified for task_type : {task_type}")
        
        return qpm_img, phase_img


class rbc_dataset(torch.utils.data.Dataset):
    '''
        A standard dataset class to get red blood cell dataset
        
        Args:
            data_dir  : data directory which contains data hierarchy as `./train/amp/00001.png`
            type_     : whether dataloader if train/ val 
            transform : torchvision.transforms
    '''
    
    def __init__(self, data_dir='/n/holyscratch01/wadduwage_lab/fypteam_22/datasets/rbc/processed/', type_= 'train', transform= None, task_type= 'phase2amp', photon_count=1, cfg= None, **kwargs):
        self.transform= transform
        self.task_type= task_type
        self.photon_count = photon_count
        self.cfg= cfg

        if type_ == 'test':
            self.phase_img_dirs = sorted(glob.glob(f'{data_dir}/test/*'), key= lambda x: x.split('/')[-1][:-4])
        else:
            self.phase_img_dirs = sorted(glob.glob(f'{data_dir}/phase/*'), key= lambda x: x.split('/')[-1][:-4])
        
        if type_ == 'train':
            self.phase_img_dirs = self.phase_img_dirs[:1000]
        elif type_ == 'val':
            self.phase_img_dirs = self.phase_img_dirs[1000:]
        
    def __len__(self):
        return len(self.phase_img_dirs)
        
    def __getitem__(self, idx): 
        phase_img = np.load(self.phase_img_dirs[idx])

        phase_img= self.transform(phase_img).to(torch.float32)
        amp_img  = torch.ones(phase_img.shape)
                
        if self.task_type=='phase2amp' or self.task_type=='phase2intensity':
            qpm_img = self.photon_count * amp_img * torch.exp(1j*phase_img)   
        else:
            raise NotImplementedError(f"Code should be verified for task_type : {task_type}")

        return qpm_img, phase_img