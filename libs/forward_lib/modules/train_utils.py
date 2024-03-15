import torch
import numpy as np
from modules.eval_metrics import *
import torchvision.transforms.functional as F
import torchvision.transforms as T
import warnings
warnings.simplefilter('default', UserWarning)


def loop(model, loader, criterion, opt, device, type_= 'train', task_type= 'phase2amp', model_type='d2nn', testing= False, cfg = None, epoch_n = None):
    '''
        Function to execute an epoch
        
        Args:
            model          : The model
            loader         : The train/ validation loader to load phase images (dtype=cfloat)
            criterion      : The loss function
            opt            : The optimizer
            device         : Device 
            type_          : The type of the loop - train or val. Defaults to 'train'
            
        Returns:
             np.mean(losses_t)  : Mean train loss (MSE) of the epoch
             np.mean(ssim11_t)  : Mean SSIM (k=11) of the epoch
             np.mean(l1_t)      : Mean L1 distance of the epoch
             ground_truth       : The last batch of ground truth images in an epoch ((n_samples, img_size, img_size), dtype=cfloat) | amp range: [0,1], phase range: [0, pi]
             pred_img           : The last batch of predicted images in an epoch ((n_samples, img_size, img_size), dtype=cfloat)
    '''
    
    losses_t= []
    ssim11_ri_t = []
    ssim11_rd_t = []
    l1_t= []

    img_size   = cfg['img_size']
    samples    = cfg['samples']
    shrinkFactor = cfg['shrink_factor'] if 'shrink_factor' in cfg.keys() else 1
    reg_schedule = cfg['reg_schedule'] if 'reg_schedule' in cfg.keys() else ['None', 0]

    if(shrinkFactor!=1):
        csize = int((img_size*samples)/shrinkFactor)
        spos  = int((img_size*samples - csize)/2)
        epos  = spos + csize
    else:
        spos = 0
        epos = img_size*samples
        

    phase_error_x  = {} #dictionary for sum(x)
    phase_error_x2 = {} #dictionary for sum(x^2)
    e_n = {}
    bin_size  = cfg['bin_size'] if 'bin_size' in cfg.keys() else 5
    angle_max = eval(cfg['angle_max']) if 'angle_max' in cfg.keys() else 2*np.pi

    for idx, (x, y) in enumerate(loader):
        if testing== True and idx>2:break
        if cfg['overfit']== True and idx>=1: break
        
        ### CLIP ANGLE TO -> [0, angle_max]
        angle_max = eval(cfg['angle_max']) #2*np.pi
        y = torch.clip(y, min= 0, max= angle_max)
        ###
        
        if cfg['error_analysis']:
            cfg['eff_neuron_size'] = cfg['neuron_size'] + cfg['dx_error'] 
            cfg['eff_layer_dist']  = cfg['delta_z'] + cfg['dz_error']
        else:
            cfg['eff_neuron_size']  =  cfg['neuron_size'] + (2 * cfg['dx_noise'] * torch.rand(1).item() - cfg['dx_noise']) #add neuron size noise N_f ~ U[-self.dx_noise, self.dx_noise]
            cfg['eff_layer_dist']   =  cfg['delta_z']     + (2 * cfg['dz_noise'] * torch.rand(1).item() - cfg['dz_noise']) #add interlayer distance noise N_f ~ U[-self.dz_noise, self.dz_noise]
            cfg['step_noise_value'] =  (2 * cfg['step_noise'] * torch.rand(1).item() - cfg['step_noise']) #phase step noise N_f ~ U[-self.dp_noise, self.dp_noise]
            # cfg['eff_phase_delta']  =  eval(cfg['phase_delta']) + cfg['step_noise_value']

        ground_truth = x[:,0].to(device) # Remove channel dimension
        if cfg['overfit']:
            ground_truth = ground_truth[22].unsqueeze(0)

        if type_=='train':
            if cfg['inq'] and epoch_n == cfg['epochs']-1 and idx>0:break # all the weights will be quantized in the final epoch (in INQ)
                
            model.train()
            opt.zero_grad()
            pred_img, out_bias,out_scale, temp, mask_list = model(ground_truth, epoch_n, idx, cfg)

            if cfg['output_scalebias_matrix']:
                out_bias= out_bias[:,spos:epos,spos:epos] 
                out_scale= out_scale[:,spos:epos,spos:epos] 

            pred_img = pred_img[:,spos:epos,spos:epos] # Crop the pred image

            if cfg['get_dataloaders'] == 'get_qpm_np_dataloaders' or cfg['get_dataloaders'] == 'get_bacteria_dataloaders':
                if cfg['unwrapped_phase']:
                    ground_truth = y[:,0].to(device)[:,spos:epos,spos:epos] / angle_max ## NORMALIZE THE RECONSTRUCTION
                    gt_angle = ground_truth
                    gt_abs = ground_truth
                else:
                    raise NotImplementedError(f"Implementation should be done st gt_angle, gt_abs in [0,1] range for SSIM calculation !!!")
            else:
                ground_truth = ground_truth[:,spos:epos,spos:epos] # Crop the groundtruth image
                gt_angle = ground_truth.angle()/np.pi
                gt_abs = ground_truth.abs()
            
            if task_type== 'phase2amp' or task_type== 'phasenoamp2amp':
                raise NotImplementedError(f"output transformation not implemented")
                loss = criterion(pred_img.abs()**2, (gt_angle)**2)
            elif task_type=='phase2intensity':
                if cfg['output_scalebias_for_INTENSITY']:
                    pred_out = out_scale * pred_img.abs()**2 + out_bias
                else:
                    warnings.warn("Warning! You are scaling the amplitude (not intensity)")
                    pred_out = (out_scale * pred_img.abs() + out_bias)**2

                if cfg['rotate_angle'] != 0:
                    gt_angle = F.rotate(gt_angle, cfg['rotate_angle'])
                if cfg['flipv']:
                    flip_transform = T.RandomVerticalFlip(p = 1)
                    gt_angle = flip_transform(gt_angle)
                if cfg['fliph']:
                    flip_transform = T.RandomHorizontalFlip(p = 1)
                    gt_angle = flip_transform(gt_angle)
                    
                if cfg['dsq'] and cfg['dsq_temp_learn']:
                    sum_sq = 0
                    for kk in range(1,cfg['n_layers']+1):
                        sum_sq += model.quant_function[kk].alpha**2
                    reg = 0.5 * cfg['dsq_regularize'][0] * (sum_sq - cfg['dsq_regularize'][1]**2)
                elif cfg['mlsq'] and cfg['learn_schedule']:
                    if reg_schedule[0] == 'pow2':
                        cur_lambda = cfg['mlsq_reg'][0] * 2**(epoch_n//reg_schedule[1])
                    elif reg_schedule[0] == 'pow1.5':
                        cur_lambda = cfg['mlsq_reg'][0] * (1.5)**(epoch_n//reg_schedule[1])
                    elif reg_schedule[0] == 'lin':
                        cur_lambda = cfg['mlsq_reg'][0] * ((epoch_n//reg_schedule[1])+1)
                    else:
                        cur_lambda = cfg['mlsq_reg'][0]
                    
                    if cfg['model'] == 'fourier_model':
                        sum_sq = model.quant_function.k**2
                    else:
                        sum_sq = 0
                        for kk in range(1,cfg['n_layers']+1):
                            sum_sq += model.quant_function[kk].k**2
                    reg = 0.5 * cur_lambda * (sum_sq - cfg['mlsq_reg'][1]**2)
                else:
                    reg = 0
                
                loss = criterion(pred_out, gt_angle) + reg
                out_complex_img = pred_out*torch.exp(1j*(pred_img.angle()))

            elif task_type== 'amp2amp':
                loss = criterion(pred_img.abs()**2, gt_abs**2)
                
            loss.backward()
            
            #### INQ
            if cfg['inq']:
                for i, param in enumerate(model.parameters()):
                    param.grad = param.grad * mask_list[i]
                    # print(epoch_n, i)
                    # if idx == 0:
                    #     print(param.grad)
            
            # opt.step()
            #### if using momentum and INQ
            if epoch_n < cfg['epochs']-1:
                opt.step()
        else:

            model.eval()
            
            with torch.no_grad():
                pred_img, out_bias,out_scale, temp, mask_list = model(ground_truth, epoch_n, idx, cfg)

                if cfg['output_scalebias_matrix']:
                    out_bias= out_bias[:,spos:epos,spos:epos] 
                    out_scale= out_scale[:,spos:epos,spos:epos] 

                pred_img = pred_img[:,spos:epos,spos:epos] # Crop the pred image

                if cfg['get_dataloaders'] == 'get_qpm_np_dataloaders' or cfg['get_dataloaders'] == 'get_bacteria_dataloaders':
                    if cfg['unwrapped_phase']:
                        gt = y[:,0].to(device)[:,spos:epos,spos:epos] / angle_max ## NORMALIZE THE RECONSTRUCTION
                        gt_angle = gt
                        gt_abs = gt
                        ground_truth = ground_truth[:,spos:epos,spos:epos].abs() + 1j*(gt*angle_max/(2*np.pi))
                    else:
                        raise NotImplementedError(f"Implementation should be sone st gt_angle, gt_abs in [0,1] range for SSIM calculation !!!")

                        # gt = ground_truth[:,spos:epos,spos:epos]
                        # gt_angle = gt.angle()%(2*np.pi)
                        # gt_abs = gt.abs()
                        # ground_truth = gt_abs + 1j*gt_angle
                else:
                    ground_truth = ground_truth[:,spos:epos,spos:epos] # Crop the groundtruth image
                    gt_angle = ground_truth.angle()/np.pi
                    gt_abs = ground_truth.abs()

                if task_type== 'phase2amp' or task_type== 'phasenoamp2amp':
                    raise NotImplementedError(f"output transformation not implemented")
                    loss = criterion(pred_img.abs()**2, (gt_angle)**2)
                    # phase_error_x, phase_error_x2, e_n =  error_quantification(gt_angle, pred_img.abs(), phase_error_x, phase_error_x2, e_n, bin_size, angle_max)
                    
                elif task_type=='phase2intensity':
                    if cfg['output_scalebias_for_INTENSITY']:
                        pred_out = out_scale * pred_img.abs()**2 + out_bias
                    else:
                        pred_out = (out_scale * pred_img.abs() + out_bias)**2
                    
                    if cfg['rotate_angle'] != 0:
                        gt_angle = F.rotate(gt_angle, cfg['rotate_angle'])
                    if cfg['flipv']:
                        flip_transform = T.RandomVerticalFlip(p = 1)
                        gt_angle = flip_transform(gt_angle)
                    if cfg['fliph']:
                        flip_transform = T.RandomHorizontalFlip(p = 1)
                        gt_angle = flip_transform(gt_angle)
                    
                    loss = criterion(pred_out, gt_angle)
                    out_complex_img = pred_out*torch.exp(1j*(pred_img.angle())) # Complex number's magnitude is the post processed intensity

                    #loss = criterion(pred_img.abs()**2, gt_angle)
                    # phase_error_x, phase_error_x2, e_n =  error_quantification(gt_angle, pred_out, phase_error_x, phase_error_x2, e_n, bin_size, angle_max)
                    
                elif task_type== 'amp2amp':
                    loss = criterion(pred_img.abs()**2, gt_abs**2)
                
             
        losses_t.append(loss.item())
            
        if task_type== 'phase2amp' or task_type== 'phasenoamp2amp':
            ssim11_ri_t.append(ssim_pytorch(pred_img.abs()**2, (gt_angle)**2, k= 11))
            ssim11_rd_t.append(ssim_pytorch(pred_img.abs()**2, (gt_angle)**2, k = 11, range_independent = False))
            l1_t.append(L1_distance(pred_img.abs()**2, (gt_angle)**2))
        elif task_type == 'phase2intensity':
            ssim11_ri_t.append(ssim_pytorch(pred_out, (gt_angle), k= 11))
            ssim11_rd_t.append(ssim_pytorch(pred_out, (gt_angle), k = 11, range_independent = False))
            l1_t.append(L1_distance(pred_out, gt_angle))
        elif task_type== 'amp2amp':
            ssim11_ri_t.append(ssim_pytorch(pred_img.abs()**2, gt_abs**2, k= 11))
            ssim11_rd_t.append(ssim_pytorch(pred_img.abs()**2, gt_abs**2, k = 11, range_independent = False))
            l1_t.append(L1_distance(pred_img.abs()**2, gt_abs**2))
            
        phase_error_x = 0 
        phase_error_x2 = 0 
        e_n = 0
                    
    return np.mean(losses_t), np.mean(ssim11_ri_t), np.mean(ssim11_rd_t), np.mean(l1_t), ground_truth, out_complex_img, phase_error_x, phase_error_x2, e_n, temp