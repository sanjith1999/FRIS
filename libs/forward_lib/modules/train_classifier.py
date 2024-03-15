import wandb
import torch
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import os
import shutil
from modules.optimizer import Lookahead

#use quantized enabled models from fabrication repo
from modules.d2nn_models_new import *
from modules.quantization import *
from modules.fourier_model import * 

#use classifier core modules from classifier repo
from classifier_core_modules.train_utils import loop
from classifier_core_modules.dataloaders import *
from classifier_core_modules.loss import *
from classifier_core_modules.vis_utils import *
from classifier_core_modules.eval_metrics import *

# from lion_pytorch import Lion

def train_and_log(cfg, model_weights = None):
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
    shrinkFactor = cfg['shrink_factor'] if 'shrink_factor' in cfg.keys() else 1
    exp_name   = cfg['exp_name']
    save_results_local = cfg['save_results_local']
    quant_func = cfg['quant_func'] if 'quant_func' in cfg.keys() else "mlsq"

    cfg['learn_schedule'] = cfg['learn_schedule'] if 'learn_schedule' in cfg.keys() else False

    train_loader, val_loader, test_loader, _, class_names = eval(cfg['get_dataloaders'])(cfg['img_size']*cfg['samples'], cfg['train_batch_size'], torch_seed, shrinkFactor =  shrinkFactor, label_type = cfg['label_type'], balanced_mode = cfg['balanced_mode'] , cfg=cfg)

    if cfg['overfit']:
        train_loader = val_loader
    
    device = cfg['device']

    torch.manual_seed(torch_seed)

    scheduler = torch.arange(cfg['schedule_start'], 1+cfg['schedule_start']+(cfg['epochs']-cfg['quant_after'])//cfg['schedule_every']*cfg['schedule_increment'], cfg['schedule_increment'])
    cfg['schedule_array'] = scheduler

    model = eval(model_type)(cfg, model_weights).to(device)
    
    if (log_wandb): wandb.watch(model)
        
    criterion= eval(cfg['loss_func'])
    # opt = RAdam(model.parameters(),lr = cfg['learning_rate']) 
    #opt        = Lookahead(base_optim, k=5, alpha=0.5)
    opt       = eval(cfg['optimizer'])(model.parameters(), lr = cfg['learning_rate'])
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,'min')

    losses_train, losses_val, losses_test = [], [], []

    print(f'exp results dir: ../results/{exp_name}')
    
    if os.path.isdir(f'../results/{exp_name}'):
        print(f'Deleting existing directory : ../results/{exp_name}')
        shutil.rmtree(f'../results/{exp_name}')
        
    os.mkdir(f'../results/{exp_name}')

    for epoch in range(cfg['epochs']):
        loss_train, ssim11_train, ssim11rd_train,l1_train,gt_img_train ,pred_img_train , accuracies_train, acc_train, weighted_f1_train, train_preds, train_labels, temp,      = loop(model, train_loader, criterion, opt, device, type_='train', task_type= task_type, model_type=model_type, testing= testing, cfg = cfg, epoch_n = epoch)
        if cfg['overfit']:
            loss_val = loss_test = loss_train
            ssim11_val = ssim11_test = ssim11_train
            ssim11rd_val = ssim11rd_test = ssim11rd_train
            l1_val = l1_test = l1_train
            gt_img_val = gt_img_test = gt_img_train
            pred_img_val = pred_img_test = pred_img_train
            accuracies_val = accuracies_test = accuracies_train
            acc_val = acc_test = acc_train
            weighted_f1_val = weighted_f1_test = weighted_f1_train
            val_preds = test_preds = train_preds
            val_labels = test_labels = train_labels 
        else:
            loss_val, ssim11_val, ssim11rd_val, l1_val, gt_img_val, pred_img_val, accuracies_val, acc_val, weighted_f1_val, val_preds, val_labels, temp,           = loop(model, val_loader  , criterion, opt, device, type_= 'val', task_type= task_type, model_type=model_type, testing= testing, cfg = cfg, epoch_n = epoch)
            loss_test, ssim11_test, ssim11rd_test, l1_test, gt_img_test, pred_img_test, accuracies_test, acc_test, weighted_f1_test, test_preds, test_labels, temp, = loop(model, test_loader , criterion, opt, device, type_= 'test', task_type= task_type, model_type=model_type, testing= testing, cfg = cfg, epoch_n = epoch)
        
        #scheduler.step(loss_val)

        losses_train.append(loss_train)
        losses_val.append(loss_val)
        losses_test.append(loss_test)
        
        
        # plot_losses(losses_val, losses_train, return_fig= True)
        
        #if epoch%save_results_local==0:
        #    plt.savefig(f'../results/{exp_name}/losses_latest.png')
        #    plt.show()

        if(epoch%save_results_local==0) or (epoch==cfg['epochs']-1):
            val_caption  = f"epoch{epoch+1}(val)@@loss_{cfg['loss_func']}({np.round(loss_val, decimals= 5)})@ssim11_ri({np.round(ssim11_val, decimals= 5)})@ssim11_rd({np.round(ssim11rd_val, decimals= 5)})@l1({np.round(l1_val, decimals= 5)})"     
            test_caption = f"epoch{epoch+1}(test)@@loss_{cfg['loss_func']}({np.round(loss_test, decimals= 5)})@ssim11_ri({np.round(ssim11_test, decimals= 5)})@ssim11_rd({np.round(ssim11rd_test, decimals= 5)})@l1({np.round(l1_test, decimals= 5)})"     
            
            if cfg['overfit']:
                val_images = plot_phase_amp_set_overfit(pred_img_val.detach().cpu(), gt_img_val.detach().cpu(), caption = val_caption, log_wandb = log_wandb, cfg = cfg)  # to plot the phase and amplitudes of a ground truth image and predicted image
                test_images = plot_phase_amp_set_overfit(pred_img_test.detach().cpu(), gt_img_test.detach().cpu(), caption = test_caption, log_wandb = log_wandb, cfg = cfg)  # to plot the phase and amplitudes of a ground truth image and predicted image
            else:
                val_images = plot_phase_amp_set(pred_img_val.detach().cpu(), gt_img_val.detach().cpu(), caption = val_caption, log_wandb = log_wandb, cfg = cfg)  # to plot the phase and amplitudes of a ground truth image and predicted image
                test_images = plot_phase_amp_set(pred_img_test.detach().cpu(), gt_img_test.detach().cpu(), caption = test_caption, log_wandb = log_wandb, cfg = cfg)  # to plot the phase and amplitudes of a ground truth image and predicted image
        
            if log_wandb:
                train_data = [[name, prec] for (name, prec) in zip(class_names, accuracies_train)]
                train_table = wandb.Table(data=train_data, columns=["class_name", "accuracy"])      
                
                val_data = [[name, prec] for (name, prec) in zip(class_names, accuracies_val)]
                val_table = wandb.Table(data=val_data, columns=["class_name", "accuracy"])   
                
                test_data = [[name, prec] for (name, prec) in zip(class_names, accuracies_test)]
                test_table = wandb.Table(data=test_data, columns=["class_name", "accuracy"])   

            train_confusion_matrix, _  = get_confusion_matrix(train_preds, train_labels, class_names)
            val_confusion_matrix, _    = get_confusion_matrix(val_preds, val_labels, class_names)
            test_confusion_matrix, _   = get_confusion_matrix(test_preds, test_labels, class_names)

            save_model_name=  f'../results/{exp_name}/latest_model_{epoch}.pth'
            torch.save({
                'state_dict': model.state_dict(),
                'cfg': cfg,
                'epoch': epoch}, save_model_name)

        # if (cfg['model'] == 'fourier_model'):
        #     images,fig = plot_phase_amp_weights_fourier(model, pred_img_val.detach().cpu(), gt_img_val.detach().cpu(), caption= caption, log_wandb = log_wandb, cfg = cfg, return_fig = True)  # to plot the phase, amplitudes of ground truth image and predicted image and the weights of the model
        # else:
            
        # images_clipped,fig_clipped = plot_phase_amp_set_clipped(pred_img_val.detach().cpu(), gt_img_val.detach().cpu(), caption= caption, log_wandb = log_wandb, cfg = cfg, return_fig = True)
        
            if cfg['mlsq'] or cfg['dsq'] or cfg['no_quant'] or cfg['hard_quant']:
                if cfg['dsq'] or cfg['mlsq'] or cfg['no_quant']:
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
            if cfg['mlsq'] or cfg['dsq'] or cfg['no_quant'] or cfg['hard_quant']:
                fig1, fig2, fig_all = generate_characteristic_curve(quant_func, quant_levels, lower_bounds, upper_bounds, cfg['learn_u'], cfg['alpha'] , temp, device, cfg)

                if log_wandb:
                    mlsq_curve = wandb.data_types.Plotly(fig1)
                    derivative_mlsq_curve = wandb.data_types.Plotly(fig2)
                    mlsq_all = wandb.data_types.Plotly(fig_all)
            else:
                mlsq_curve, derivative_mlsq_curve,mlsq_all   = None, None, None
            
            if log_wandb:
                train_bar_plot = wandb.plot.bar(train_table, "class_name" , "accuracy", title="Train Per Class Accuracy")
                val_bar_plot   = wandb.plot.bar(val_table, "class_name" , "accuracy", title="Val Per Class Accuracy")
                test_bar_plot  = wandb.plot.bar(test_table, "class_name" , "accuracy", title="test Per Class Accuracy")
                    
        else:
            images = None

            train_bar_plot, val_bar_plot, test_bar_plot  = None, None, None
            train_confusion_matrix, val_confusion_matrix, test_confusion_matrix  = None, None, None
            mlsq_curve, derivative_mlsq_curve, mlsq_all   = None, None, None

        if (log_wandb): 
            wandb.log({
                "Training Loss": loss_train,
                "Training SSIM11_RI":ssim11_train,
                "Training SSIM11_RD": ssim11rd_train,
                "Training L1":l1_train,
                "Training Acc": acc_train,
                "Training w. F1" : weighted_f1_train,
                "train class accuracies": train_bar_plot,
                "train_confusion_matrix" : train_confusion_matrix,

                "Validation Loss": loss_val,
                "Validation SSIM11_RI":ssim11_val,
                "Validation SSIM11_RD": ssim11rd_val,
                "Validation L1":l1_val,
                "Validation Acc": acc_val,
                "Validation w. F1" : weighted_f1_val,
                "Validation Examples": val_images,
                "val class accuracies": val_bar_plot,
                "val_confusion_matrix" : val_confusion_matrix,
                
                "Test Loss": loss_test,
                "Test SSIM11_RI":ssim11_test,
                "Test SSIM11_RD": ssim11rd_test,
                "Test L1":l1_test,
                "Test Acc": acc_test,
                "Test w. F1" : weighted_f1_test,
                "Test Examples": test_images,
                "test class accuracies": test_bar_plot,
                "test_confusion_matrix" : test_confusion_matrix,
                
                
                
                # "Clipped Validation Examples":images_clipped,

                "Slope Factor" : temp,

                "mlsq curve": mlsq_curve,
                "derivative of mlsq": derivative_mlsq_curve,
                "mlsq all": mlsq_all,

                "epoch":epoch+1})
        
        # save_model_name=  f'../results/{exp_name}/latest_model_{epoch}.pth'
        # torch.save({
        #     'state_dict': model.state_dict(),
        #     'cfg': cfg,
        #     'epoch': epoch}, save_model_name)
