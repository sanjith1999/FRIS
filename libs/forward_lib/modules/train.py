import wandb
from modules.train_utils import loop
from modules.dataloaders import *
from modules.d2nn_models import d2nn
from modules.d2nn_models_new import *
from modules.fourier_model import *
from modules.quantization import *

import torch
import matplotlib.pyplot as plt
from modules.vis_utils import *
from torch import nn
import numpy as np
from modules.loss import *
import os
import shutil


def train_and_log(cfg, model_weights= None):
    '''
        Function to train and log results to wandb
        
        Args:
            cfg: The dictionary containing all the required configurations
    '''
    
    
    log_wandb  = cfg['log_wandb']
    torch_seed = cfg['torch_seed']
    task_type  = cfg['task_type']
    model_type = cfg['model']
    testing    = cfg['testing']
    biasOnoise = cfg['biasOnoise']
    shrinkFactor = cfg['shrink_factor'] if 'shrink_factor' in cfg.keys() else 1
    exp_name   = cfg['exp_name']
    save_results_local = cfg['save_results_local']
    photon_count = cfg['photon_count'] if 'photon_count' in cfg.keys() else 1 ## SHOULD BE FIXED
    quant_func = cfg['quant_func'] if 'quant_func' in cfg.keys() else "mlsq"
    
    cfg['learn_schedule'] = cfg['learn_schedule'] if 'learn_schedule' in cfg.keys() else False

    train_loader, val_loader, _ = eval(cfg['get_dataloaders'])(cfg['img_size']*cfg['samples'], cfg['train_batch_size'], torch_seed,  task_type= task_type, shrinkFactor = shrinkFactor, biasOnoise = biasOnoise, photon_count=photon_count, cfg= cfg)
    device = cfg['device']
    
    if cfg['overfit']: train_loader = val_loader
    

    torch.manual_seed(torch_seed)
    # scheduler = torch.arange(cfg['schedule_start'], cfg['schedule_start']+cfg['epochs']//cfg['schedule_every']*cfg['schedule_increment'], cfg['schedule_increment'])
    scheduler = torch.arange(cfg['schedule_start'], 1+cfg['schedule_start']+(cfg['epochs']-cfg['quant_after'])//cfg['schedule_every']*cfg['schedule_increment'], cfg['schedule_increment'])
    cfg['schedule_array'] = scheduler

    model = eval(model_type)(cfg, model_weights).to(device)
    
    if (log_wandb): wandb.watch(model)
        
    criterion= eval(cfg['loss_func'])
    opt= torch.optim.Adam(model.parameters(), lr= cfg['learning_rate'])
    # opt= torch.optim.Adam(model.parameters(), lr= cfg['learning_rate'], betas=(0,0)) # use this if using INQ

    losses_train, losses_val = [], []
    
    print(f'exp results dir: ../results/{exp_name}')
    
    if os.path.isdir(f'../results/{exp_name}'):
        print(f'Deleting existing directory : ../results/{exp_name}')
        shutil.rmtree(f'../results/{exp_name}')
        
    os.mkdir(f'../results/{exp_name}')

    for epoch in range(cfg['epochs']):
        mse_loss_train, ssim11_ri_train, ssim11_rd_train, l1_train,_ , _, _, _, _, temp         = loop(model, train_loader, criterion, opt, device, type_='train', task_type= task_type, model_type=model_type, testing= testing, cfg = cfg, epoch_n = epoch)
        mse_loss_val, ssim11_ri_val, ssim11_rd_val, l1_val, gt_img_val, pred_img_val, phase_error_x, phase_error_x2, e_n, _ = loop(model, val_loader, criterion, opt, device, type_= 'val', task_type= task_type, model_type=model_type, testing= testing, cfg = cfg, epoch_n = epoch)

        
        losses_train.append(mse_loss_train)
        losses_val.append(mse_loss_val)
        
        # plot_losses(losses_val, losses_train, return_fig= True)
        
        # if epoch%save_results_local==0:
        #     print('saving loss curve ...')
        #     plt.savefig(f'../results/{exp_set_name}/{exp_name}/losses_latest.png')
        #     plt.show()
        if(epoch%save_results_local==0) or (epoch==cfg['epochs']-1):

            caption = f"epoch{epoch+1}(val)@@loss_mse({np.round(mse_loss_val, decimals= 5)})@ssim11_ri({np.round(ssim11_ri_val, decimals= 5)})@ssim11_rd({np.round(ssim11_rd_val, decimals= 5)})@l1({np.round(l1_val, decimals= 5)})"
        
        # images, fig = plot_phase_amp_set(pred_img_val.detach().cpu(), gt_img_val.detach().cpu(), caption= caption, log_wandb = log_wandb, return_fig= True)  # to plot the phase and amplitudes of a ground truth image and predicted image
        
            if cfg['overfit']:
                images = plot_single_example(pred_img_val.detach().cpu(), gt_img_val.detach().cpu(), caption= caption, log_wandb = log_wandb, cfg = cfg)
                images_ = images
            elif (cfg['model'] == 'd2nn_fourier'):
                images = plot_phase_amp_weights_fourier(model, pred_img_val.detach().cpu(), gt_img_val.detach().cpu(), caption= caption, log_wandb = log_wandb, cfg = cfg)  # to plot the phase, amplitudes of ground truth image and predicted image and the weights of the model
            else:
                images = plot_phase_amp_set(pred_img_val.detach().cpu(), gt_img_val.detach().cpu(), caption= caption, log_wandb = log_wandb, cfg = cfg)  # to plot the phase and amplitudes of a ground truth image and predicted image
                images_ = plot_phase_amp_set_flex(pred_img_val.detach().cpu(), gt_img_val.detach().cpu(), caption= caption, log_wandb = log_wandb, cfg = cfg)  # to plot the phase and amplitudes of a ground truth image and predicted image
            ## Plot Quantitative Phase Error Distribution of Validation Set
            # phase_error_dist, angle_bin_count, mean_phase_error_in_degrees = plot_phase_error_distribution(phase_error_x, phase_error_x2, e_n, log_wandb = log_wandb)
            phase_error_dist = 0 
            angle_bin_count = 0
            mean_phase_error_in_degrees = 0
        
        
            if cfg['mlsq'] or cfg['dsq'] or cfg['no_quant'] or cfg['hard_quant']:
                if (cfg['model'] != 'fourier_model' and (cfg['dsq'] or cfg['mlsq'] or cfg['no_quant'])):
                    upper_bounds = [m.detach().item() for m in model.quant_function[1].u] if isinstance(model.quant_function[1].u,  nn.ParameterList) else model.quant_function[1].u
                    lower_bounds = model.quant_function[1].l
                    quant_levels = [m for m in model.quant_function[1].n] if isinstance(model.quant_function[1].n, list) else model.quant_function[1].n
                else:
                    upper_bounds = [m.detach().item() for m in model.quant_function.u] if isinstance(model.quant_function.u,  nn.ParameterList) else model.quant_function.u
                    lower_bounds = model.quant_function.l
                    quant_levels = [m for m in model.quant_function.n] if isinstance(model.quant_function.n, list) else model.quant_function.n
            else:
                upper_bounds = cfg['upper_bounds']

            #### Get characteristic curves
            # if cfg['no_quant']:
            #     fig1 = None
            #     fig2 = None
            #     fig_all = None
            # else:
            if cfg['mlsq'] or cfg['dsq'] or cfg['no_quant'] or cfg['hard_quant']:
                fig1, fig2, fig_all = generate_characteristic_curve(quant_func, quant_levels, lower_bounds, upper_bounds, cfg['learn_u'], cfg['alpha'] , temp, device, cfg)

                # temp = temp.detach()

                mlsq_curve = wandb.data_types.Plotly(fig1)
                derivative_mlsq_curve = wandb.data_types.Plotly(fig2)

                mlsq_all = wandb.data_types.Plotly(fig_all)
            else:
                mlsq_curve, derivative_mlsq_curve,mlsq_all   = None, None, None

        else:
            images = None

            mlsq_curve, derivative_mlsq_curve            = None, None

            mlsq_all = None

        if (log_wandb): 
            wandb.log({"Training MSE Loss": mse_loss_train,
                "Training SSIM11_RI":ssim11_ri_train,
                "Training SSIM11_RD":ssim11_rd_train,
                "Training L1":l1_train,

                "Validation MSE Loss": mse_loss_val,
                "Validation SSIM11_RI":ssim11_ri_val,
                "Validation SSIM11_RD":ssim11_rd_val,
                "Validation L1":l1_val,

                "Validation Examples": images,
                "Validation Examples Flex": images_,
                "Phase Error Distribution " : phase_error_dist,
                "Angle Bin Count" : angle_bin_count,
                "Mean Phase Error (Degrees)" : mean_phase_error_in_degrees,
                "Current Temperature Factor" : temp,

                "mlsq curve": mlsq_curve,
                "derivative of mlsq": derivative_mlsq_curve,
                
                "mlsq all": mlsq_all,

                "epoch":epoch+1})
        
            save_model_name=  f'../results/{exp_name}/latest_model_{epoch}.pth'
            torch.save({
                'state_dict': model.state_dict(),
                'cfg': cfg,
                'epoch': epoch}, save_model_name)
        
            #wandb.save(f'{exp_name}.pth', base_path='../saved_models/')
            #wandb.save(save_model_name)

            #os.remove(save_model_name)
    
