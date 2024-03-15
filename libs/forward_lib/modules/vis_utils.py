import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn.functional as F
import wandb
from torchvision.utils import make_grid
import cv2
import math


from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_phase_amp(pred_img, gt_img, caption= 'no caption', log_wandb= False):
    '''
        Function to plot phases and amplitudes of ground truth and predicted complex images

            Args:
                pred_img   : Predicted image ((img_size, img_size), dtype= torch.cfloat)
                gt_img     : Ground truth image ((img_size, img_size), dtype= torch.cfloat)
                caption    : Caption for logging and titles
                log_wandb  : Whether logging will be done | bool

            Returns:
                images    : images to feed wandb logging | wandb.Image   
    '''

    
    
    fig = plt.figure(figsize= (10,10))
    plt.subplot(2,2,1)
    plt.imshow(gt_img.angle().detach().cpu().numpy(), vmin = 0, vmax= np.pi)
    plt.colorbar()
    plt.title('Ground Truth : Phase')
    plt.subplot(2,2,2)
    plt.imshow(gt_img.abs().detach().cpu().numpy(), vmin=0, vmax= 1)
    plt.colorbar()
    plt.title('Ground Truth : Amplitude')
    plt.subplot(2,2,3)
    plt.imshow(pred_img.detach().angle().cpu().numpy(), vmin= 0, vmax= np.pi)
    plt.colorbar()
    plt.title("Reconstructed : Phase")
    plt.subplot(2,2,4)
    plt.imshow(pred_img.detach().abs().cpu().numpy(), vmin= 0, vmax= 1)
    plt.colorbar()
    plt.title("Reconstructed : Amplitude")
    plt.suptitle(caption)
    plt.show()
    

    ## Wandb
    if log_wandb:
        images = wandb.Image(fig  , caption=caption)
    else:images= None
    
    plt.cla()
    plt.close(fig)
    
    return images

def plot_single_example(pred_img_set, gt_img_set, caption= 'no caption', log_wandb= False, cfg = None):
    gt_abs = torch.abs(gt_img_set)[0]
    gt_angle = torch.angle(gt_img_set)[0]
    pred_abs = torch.abs(pred_img_set)[0]
    pred_angle = torch.angle(pred_img_set)[0]%(2*np.pi)
    
    fig = plt.figure(figsize= (9.5,11))
    plt.subplot(3,2,1)
    plt.imshow(gt_angle, vmin =  0, vmax= 1)
    plt.colorbar()
    plt.title('Ground Truth : Phase')
    
    plt.subplot(3,2,2)
    plt.imshow(gt_abs, vmin =  0, vmax= 1)
    plt.colorbar()
    plt.title('Ground Truth : Amplitude')

    plt.subplot(3,2,3)
    plt.imshow(pred_angle, vmin =  0, vmax= 2*np.pi)
    plt.colorbar()
    plt.title("Reconstructed : Phase")
    
    plt.subplot(3,2,4)
    plt.imshow(pred_abs**0.5, vmin= 0, vmax= 1)
    plt.colorbar()
    plt.title("Reconstructed : Amplitude")
    
    plt.subplot(3,2,5)
    plt.imshow(pred_abs, vmin= 0, vmax= 1)
    plt.colorbar()
    plt.title("Reconstructed : Intensity")
    
    plt.suptitle(caption)
    plt.show()
    

    ## Wandb
    if log_wandb:
        images = wandb.Image(fig  , caption=caption)
    else:images= None
    
    plt.cla()
    plt.close(fig)
    
    return images

def plot_phase_amp_set(pred_img_set, gt_img_set, caption= 'no caption', log_wandb= False, cfg = None):    
    pred_img_set= pred_img_set.unsqueeze(dim= 1)[0:4]
    gt_img_set= gt_img_set.unsqueeze(dim= 1)[0:4]
    
    if cfg['get_dataloaders'] == 'get_qpm_np_dataloaders' or cfg['get_dataloaders'] == 'get_bacteria_dataloaders':
        gt_angle = gt_img_set.detach().cpu().imag
        gt_abs = gt_img_set.detach().cpu().real
    else:
        gt_angle = gt_img_set.angle().detach().cpu()
        gt_abs = gt_img_set.abs().detach().cpu()
        
    fig = plt.figure(figsize= (9.5,11))
    plt.subplot(3,2,1)
    plt.imshow(cv2.cvtColor(make_grid(gt_angle, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY),vmin =  0, vmax= 1)
    plt.colorbar()
    plt.title('Ground Truth : Phase')
    
    plt.subplot(3,2,2)
    plt.imshow(cv2.cvtColor(make_grid(gt_abs, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY),vmin =  0, vmax= 1)
    plt.colorbar()
    plt.title('Ground Truth : Amplitude')
    
    pred_angle = pred_img_set.angle().detach().cpu()%(2*np.pi)

    plt.subplot(3,2,3)
    plt.imshow(cv2.cvtColor(make_grid(pred_angle, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY), vmin =  0, vmax= 2*np.pi)
    plt.colorbar()
    plt.title("Reconstructed : Phase")
    
    plt.subplot(3,2,4)
    plt.imshow(cv2.cvtColor(make_grid(pred_img_set.abs().detach().cpu()**0.5, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY), vmin= 0, vmax= 1)
    plt.colorbar()
    plt.title("Reconstructed : Amplitude")
    
    plt.subplot(3,2,5)
    plt.imshow(cv2.cvtColor(make_grid(pred_img_set.abs().detach().cpu(), nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY), vmin= 0, vmax= 1)
    plt.colorbar()
    plt.title("Reconstructed : Intensity")
    
    plt.suptitle(caption)
    plt.show()
    

    ## Wandb
    if log_wandb:
        images = wandb.Image(fig  , caption=caption)
    else:images= None
    
    plt.cla()
    plt.close(fig)
    
    return images


def plot_phase_amp_set_flex(pred_img_set, gt_img_set, caption= 'no caption', log_wandb= False, cfg = None):    
    pred_img_set= pred_img_set.unsqueeze(dim= 1)[0:4]
    gt_img_set= gt_img_set.unsqueeze(dim= 1)[0:4]
    
    if cfg['get_dataloaders'] == 'get_qpm_np_dataloaders' or cfg['get_dataloaders'] == 'get_bacteria_dataloaders':
        gt_angle = gt_img_set.detach().cpu().imag
        gt_abs = gt_img_set.detach().cpu().real
    else:
        gt_angle = gt_img_set.angle().detach().cpu()
        gt_abs = gt_img_set.abs().detach().cpu()
        
    fig = plt.figure(figsize= (9.5,11))
    plt.subplot(3,2,1)
    plt.imshow(cv2.cvtColor(make_grid(gt_angle, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title('Ground Truth : Phase')
    
    plt.subplot(3,2,2)
    plt.imshow(cv2.cvtColor(make_grid(gt_abs, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title('Ground Truth : Amplitude')
    
    pred_angle = pred_img_set.angle().detach().cpu()%(2*np.pi)

    plt.subplot(3,2,3)
    plt.imshow(cv2.cvtColor(make_grid(pred_angle, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title("Reconstructed : Phase")
    
    plt.subplot(3,2,4)
    plt.imshow(cv2.cvtColor(make_grid(pred_img_set.abs().detach().cpu()**0.5, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title("Reconstructed : Amplitude")
    
    plt.subplot(3,2,5)
    plt.imshow(cv2.cvtColor(make_grid(pred_img_set.abs().detach().cpu(), nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title("Reconstructed : Intensity")
    
    plt.suptitle(caption)
    plt.show()
    

    ## Wandb
    if log_wandb:
        images = wandb.Image(fig  , caption=caption)
    else:images= None
    
    plt.cla()
    plt.close(fig)
    
    return images

def plot_phase_amp_weights_fourier(model, pred_img_set, gt_img_set, caption= 'no caption', log_wandb= False, cfg = None):    
    pred_img_set= pred_img_set.unsqueeze(dim= 1)[10:14]
    gt_img_set= gt_img_set.unsqueeze(dim= 1)[10:14]
    n_layers = len(model.layer_blocks)
    add_rows = math.ceil((n_layers*3)/6)
    
    if cfg['get_dataloaders'] == 'get_qpm_np_dataloaders' or cfg['get_dataloaders'] == 'get_bacteria_dataloaders':
        gt_angle = gt_img_set.detach().cpu().imag
        gt_abs = gt_img_set.detach().cpu().real
    else:
        gt_angle = gt_img_set.angle().detach().cpu()
        gt_abs = gt_img_set.abs().detach().cpu()
        
    fig = plt.figure(figsize= (25,8))
    plt.subplot(1+add_rows,6,1)
    plt.imshow(cv2.cvtColor(make_grid(gt_angle, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title('Ground Truth : Phase')
    
    plt.subplot(1+add_rows,6,2)
    plt.imshow(cv2.cvtColor(make_grid(gt_abs, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title('Ground Truth : Amplitude')
    
    plt.subplot(1+add_rows,6,3)
    plt.imshow(cv2.cvtColor(make_grid(pred_img_set.angle().detach().cpu(), nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title("Reconstructed : Phase")
    
    plt.subplot(1+add_rows,6,4)
    plt.imshow(cv2.cvtColor(make_grid(pred_img_set.abs().detach().cpu(), nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title("Reconstructed : Amplitude")
    
    plt.subplot(1+add_rows,6,5)
    plt.imshow(cv2.cvtColor(make_grid(pred_img_set.abs().detach().cpu()**2, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title("Reconstructed : Intensity")
    
    for idx in range(n_layers):
        ts_amp = torch.sigmoid(model.layer_blocks[idx].amp_weights.detach().cpu())
        ts_phase = model.layer_blocks[idx].phase_weights.detach().cpu()

        plt.subplot(1+add_rows,6,7+(3*idx))
        plt.imshow(ts_phase.numpy())
        plt.colorbar()
        plt.title(f"t (Unwrapped Phase) : Layer{idx}")
        
        plt.subplot(1+add_rows,6,7+(3*idx)+1)
        plt.imshow(ts_phase.numpy()%(2*np.pi))
        plt.colorbar()
        plt.title(f"t (Wrapped Phase) : Layer{idx}")

        plt.subplot(1+add_rows,6,7+(3*idx)+2)
        plt.imshow(ts_amp.numpy())
        plt.colorbar()
        plt.title(f"t (Amplitude) : Layer{idx}")
            
    
    plt.suptitle(caption)
    plt.show()
    

    ## Wandb
    if log_wandb:
        images = wandb.Image(fig  , caption=caption)
    else:images= None
    
    plt.cla()
    plt.close(fig)
    
    return images

def plot_complex_image(img, caption= 'no caption'):
    '''
        Function to plot phases and amplitudes a complex image
            Args:
                img: complex image ((img_size, img_size), torch.cfloat) | torch.Tensor
    '''
    
    fig = plt.figure(figsize= (10,7))
    plt.subplot(2,2,1)
    plt.imshow(img.angle().detach().cpu().numpy(), vmin= 0)
    plt.colorbar()
    plt.title('Phase')
    plt.subplot(2,2,2)
    plt.imshow(img.abs().detach().cpu().numpy(), vmin=0)
    plt.colorbar()
    plt.title('Amplitude')
    plt.suptitle(caption)
    plt.show()

def plot_losses(losses_val, losses_train, return_fig= False):
    '''
        Function to plot losses
            Args:
                losses_val   : Validation losses of each epoch | list
                losses_train : Train losses of each epoch | list
    '''


    plt.figure()
    plt.plot(losses_val, label= 'val loss')
    plt.plot(losses_train, label= 'train loss')
    plt.legend()
    plt.title('losses')
    
    if not return_fig:plt.show()
    
    
def advance_through_network(model, gt_img):
    '''
        Function to show outputs of each layer 
            Args:
                model   : D2NN model
                gt_img  : Ground truth image ((n_samples, img_size, img_size), dtype= torch.cfloat)
    '''    
    
    out1 = model.input_layer(gt_img)
    plot_amp_and_phase((out1.detach().cpu().numpy())[0], gt_img[0], caption= 'At the input plane')

    out2 = model.layer1(out1)
    plot_amp_and_phase((out2.detach().cpu().numpy())[0], gt_img[0], caption= "Layer1 output")

    out3 = model.layer2(out2)
    plot_amp_and_phase((out3.detach().cpu().numpy())[0], gt_img[0], caption= "Layer2 output")

    out4 = model.layer3(out3)
    plot_amp_and_phase((out4.detach().cpu().numpy())[0], gt_img[0], caption= "Layer3 output")

    out5 = model.layer4(out4)
    plot_amp_and_phase((out5.detach().cpu().numpy())[0], gt_img[0], caption= "Layer4 output")

    out6 = model.layer5(out5)
    plot_amp_and_phase((out6.detach().cpu().numpy())[0], gt_img[0], caption= "Layer5 (Final) output")
    

def plot_transmission_coef(model):
    '''
        Function to visualize the transmission coefficients 
            Args:
                model   : D2NN model
    '''   
    
    plot_complex_image(model.layer1.ts.detach().cpu(), caption= 'Transmission coefficients: Layer1')
    plot_complex_image(model.layer2.ts.detach().cpu(), caption= 'Transmission coefficients: Layer2')
    plot_complex_image(model.layer3.ts.detach().cpu(), caption= 'Transmission coefficients: Layer3')
    plot_complex_image(model.layer4.ts.detach().cpu(), caption= 'Transmission coefficients: Layer4')
    plot_complex_image(model.layer5.ts.detach().cpu(), caption= 'Transmission coefficients: Layer5')

def plot_phase_error_distribution(phase_error_x, phase_error_x2, e_n, bin_size = 5,  txt  = "Val ", log_wandb = False):
    keys = phase_error_x.keys()

    error_mean = []
    error_std  = []


    for key in phase_error_x.keys():
        
        error_mean.append(phase_error_x[key]/e_n[key]) # sum of bin/ number of samples
        try:
            std_ =  math.sqrt(phase_error_x2[key]/e_n[key] - error_mean[-1]**2)
        except:
            print("STD error! in bin " +str(key) + " n_samples " + str(e_n[key]) + " sum of bin " + str(phase_error_x[key]) + " sum of bin squared " + str(phase_error_x2[key]))
            std_ = 0
        error_std.append(std_)

    mean_phase_error_in_degrees = np.sum(list(phase_error_x.values()))/np.sum(list(e_n.values()))

    ## Error Plot
    fig1 = plt.figure(figsize=(15,3))
    plt.title(f"Mean Phase Error (degrees) = {mean_phase_error_in_degrees}")
    caption1 = txt +  " Phase Error Distribution"
    upper_bound = np.max(error_mean)+ np.max(error_std) + 5

    plt.bar(keys, error_mean, yerr = error_std, label= caption1, width = bin_size*0.8)
    plt.ylim(0,int(upper_bound))
    plt.ylabel ('Mean L1 Distance (°)')
    plt.xlabel ('Angle (degrees)')
    plt.xticks(list(keys)*2, rotation= 90)
    plt.legend (bbox_to_anchor=(1, 1), loc="upper right", borderaxespad=0.)
    plt.show()

        
    ## Count of angle bins
    fig2 = plt.figure(figsize=(15,3))
    total_pixels = np.sum(list(e_n.values()))
    plt.title(f"Total pixels = {total_pixels}")
    caption2 = txt  + "Angle Bin Count"
    plt.bar(keys, list(e_n.values())/total_pixels, label= caption2, width = bin_size*0.8,  color=(0.2, 0.2, 0.2, 0.6))
    plt.ylabel ('Count')
    plt.xlabel ('Angle (degrees)')
    plt.xticks(list(keys)*2, rotation= 90)
    plt.legend (bbox_to_anchor=(1, 1), loc="upper right", borderaxespad=0.)
    plt.show()


    ## Wandb
    if log_wandb:
        image1 = wandb.Image(fig1, caption = caption1)
        image2 = wandb.Image(fig2, caption = caption2)
    else:
        image1 = None
        image2 = None
    
    return image1, image2, mean_phase_error_in_degrees