import torch
import numpy as np
from torch import nn
from libs.forward_lib.modules.diffraction import *
from libs.forward_lib.modules.d2nn_layers import *
from libs.forward_lib.modules.saturable_absorption import *
from libs.forward_lib.modules.quantization import *
import warnings
warnings.simplefilter('default', UserWarning)

class d2nnASwWindow(nn.Module):
    '''
        Diffractive Deep Neural Network
    '''
    def __init__(self, cfg, weights = None):
        super(d2nnASwWindow, self).__init__()
        
        self.n_i = cfg['img_size']
        self.n_o= cfg['img_size']

        self.delta_z = cfg['delta_z']
        self.lambda_ = cfg['lambda_']
        self.int_ramp= cfg['int_ramp']
        self.neuron_size= cfg['neuron_size']
        self.learn_type= cfg['learn_type']
        self.device= cfg['device']
        self.n_layers= cfg['n_layers']
        self.energy_type = cfg['energy_type']
        self.in_dist  = cfg['in_dist']
        self.out_dist = cfg['out_dist']
        self.window_size= cfg['window_size']
        self.in_dist= cfg['in_dist']
        self.samples = cfg['samples']

        self.init_weights = weights
        self.RI_Gel  = cfg['RI_Gel']
        self.gs_quant = cfg['gs_quant'] if 'gs_quant' in cfg.keys() else False
        
        n_hidden= (self.n_i+ self.n_o)//2
        
        if self.gs_quant:
            d2nn_layer = d2nn_GS_layer
        else:
            d2nn_layer= d2nnASwWindow_layer
        
        self.quant_function = nn.ModuleList()
        for i in range(self.n_layers+1):
            if cfg['dsq']:
                self.quant_function.append(dsq(cfg['quant_levels'], cfg['lower_bound'], cfg['upper_bounds'], cfg['learn_u'], cfg['dsq_alpha']))
            else:
                self.quant_function.append(mlsq(cfg['quant_levels'], cfg['lower_bound'], cfg['upper_bounds'], cfg['learn_u'], cfg['alpha']))
  
        self.layer_blocks: nn.ModuleList[d2nn_layer] = nn.ModuleList()
        #self.layer_blocks.append(d2nn_layer(self.n_i, n_hidden, self.delta_z, self.lambda_, neuron_size= self.neuron_size, learn_type='no', device= self.device))
        self.layer_blocks.append(d2nn_layer(self.n_i, n_hidden, self.in_dist, self.lambda_, neuron_size= self.neuron_size, learn_type='no', device= self.device, window_size= self.window_size, RI = self.RI_Gel, cfg = cfg, quant_func = self.quant_function[0]))

        for idx in range(self.n_layers-1):
            if weights != None:
                weight_dict = {'amp_weights':torch.ones((self.n_i, self.n_i), dtype= torch.float), 'phase_weights': self.init_weights[f'layer_blocks.{idx+1}.phase_weights']}
                self.layer_blocks.append(d2nn_layer(n_hidden, n_hidden, self.delta_z, self.lambda_, neuron_size= self.neuron_size, learn_type=self.learn_type, device= self.device, energy_type = self.energy_type, window_size= self.window_size, RI = self.RI_Gel, weights= weight_dict, cfg = cfg, quant_func = self.quant_function[idx+1]))
            else:
                self.layer_blocks.append(d2nn_layer(n_hidden, n_hidden, self.delta_z, self.lambda_, neuron_size= self.neuron_size, learn_type=self.learn_type, device= self.device, energy_type = self.energy_type, window_size= self.window_size, RI = self.RI_Gel, cfg = cfg, quant_func = self.quant_function[idx+1]))
            
        if weights != None:
            weight_dict = {'amp_weights':torch.ones((self.n_i, self.n_i), dtype= torch.float), 'phase_weights': self.init_weights[f'layer_blocks.{self.n_layers}.phase_weights']}
            self.layer_blocks.append(d2nn_layer(n_hidden, self.n_o, self.out_dist, self.lambda_, neuron_size= self.neuron_size, learn_type=self.learn_type, device= self.device, energy_type = self.energy_type, window_size= self.window_size, RI = self.RI_Gel, weights= weight_dict, cfg = cfg, quant_func = self.quant_function[-1]))
        else:
            self.layer_blocks.append(d2nn_layer(n_hidden, self.n_o, self.out_dist, self.lambda_, neuron_size= self.neuron_size, learn_type=self.learn_type, device= self.device, energy_type = self.energy_type, window_size= self.window_size,RI = self.RI_Gel, cfg = cfg, quant_func = self.quant_function[-1]))
        
        if cfg['output_scalebias_matrix']:
            warnings.warn("Warning! You are attempting to use a Matrix to scale outputs !")
            
            if cfg['output_scale'] == 'randn':
                output_scale = torch.randn(1, self.n_o, self.n_o)
            else:
                output_scale = cfg['output_scale'] * torch.ones(1, self.n_o, self.n_o)
                
            if cfg['output_bias'] == 'randn':
                output_bias = torch.randn(1, self.n_o, self.n_o)
            else:
                output_bias = cfg['output_bias'] * torch.ones(1, self.n_o, self.n_o)
        else:
            output_scale=torch.tensor(cfg['output_scale'])
            output_bias= torch.tensor(cfg['output_bias'])
            
        if cfg['output_scale_learnable']:
            self.output_scale= nn.Parameter(output_scale)
        else:
            self.output_scale= output_scale
            
        if cfg['output_bias_learnable']:
            self.output_bias= nn.Parameter(output_bias)
        else:
            self.output_bias= output_bias

    def forward(self, input_e_field, epoch_n, batch_iter = None, cfg = None):
        '''
            Function for forward pass

            Args:
                input_e_field  : Input electric field (batch_size, self.n_i, self.n_i)
            Returns:
                output_e_field : Output E-field of D2NN
        '''
        x= input_e_field.view(-1, self.n_i*self.samples, self.n_i*self.samples)
        device = input_e_field.device
        
        mask_list = []   ### masks for each layer (INQ)
        for idx in range(len(self.layer_blocks)):
            x, temp, mask = self.layer_blocks[idx](x,epoch_n, batch_iter, cfg)
            if idx > 0:
                mask_list.append(mask)
                
        return x, self.output_bias.to(device), self.output_scale.to(device), temp, mask_list
    
      
class d2nn_non_linear(nn.Module):
    '''
        Diffractive Deep Neural Network with non-linearities
    '''
    def __init__(self, cfg, weights = None):
        super(d2nn_non_linear, self).__init__()
        
        self.n_i = cfg['img_size']
        self.n_o= cfg['img_size']

        self.delta_z = cfg['delta_z']
        self.lambda_ = cfg['lambda_']
        self.int_ramp= cfg['int_ramp']
        self.neuron_size= cfg['neuron_size']
        self.learn_type= cfg['learn_type']
        self.device= cfg['device']
        self.n_layers= cfg['n_layers']
        self.energy_type = cfg['energy_type']
        self.in_dist  = cfg['in_dist']
        self.out_dist = cfg['out_dist']
        self.window_size= cfg['window_size']
        self.in_dist= cfg['in_dist']
        self.samples = cfg['samples']

        self.init_weights = weights
        self.RI_Gel  = cfg['RI_Gel']
        
        self.init_abs   = cfg['init_abs']
        self.max_int    = cfg['max_int'] if 'max_int' in cfg.keys() else 1e8
        self.poly_order = cfg['poly_order'] if 'poly_order' in cfg.keys() else 15
        self.nll_idx    = cfg['nll_idx']
        
        n_hidden= (self.n_i+ self.n_o)//2
        d2nn_layer= d2nnASwWindow_layer
        nl_layer = sat_abs_layer
  
        self.layer_blocks: nn.ModuleList[d2nn_layer] = nn.ModuleList()
        #self.layer_blocks.append(d2nn_layer(self.n_i, n_hidden, self.delta_z, self.lambda_, neuron_size= self.neuron_size, learn_type='no', device= self.device))
        self.layer_blocks.append(d2nn_layer(self.n_i, n_hidden, self.in_dist, self.lambda_, neuron_size= self.neuron_size, learn_type='no', device= self.device, window_size= self.window_size, RI = self.RI_Gel, cfg = cfg))

        for idx in range(self.n_layers-1):
            if idx in self.nll_idx:
                self.layer_blocks.append(nl_layer(n_hidden, n_hidden, self.delta_z, self.lambda_, self.neuron_size, init_abs= self.init_abs, window_size= self.window_size, RI = self.RI_Gel, n= self.poly_order, device = self.device, cfg = cfg, max_intensity = self.max_int, data_path = "saturable_absorption/data"))
            elif weights != None:
                weight_dict = {'amp_weights':torch.ones((self.n_i, self.n_i), dtype= torch.float), 'phase_weights': self.init_weights[f'layer_blocks.{idx+1}.phase_weights']}
                self.layer_blocks.append(d2nn_layer(n_hidden, n_hidden, self.delta_z, self.lambda_, neuron_size= self.neuron_size, learn_type=self.learn_type, device= self.device, energy_type = self.energy_type, window_size= self.window_size, RI = self.RI_Gel, weights= weight_dict, cfg = cfg))
            else:
                self.layer_blocks.append(d2nn_layer(n_hidden, n_hidden, self.delta_z, self.lambda_, neuron_size= self.neuron_size, learn_type=self.learn_type, device= self.device, energy_type = self.energy_type, window_size= self.window_size, RI = self.RI_Gel,  cfg = cfg))
            
        if weights != None:
            weight_dict = {'amp_weights':torch.ones((self.n_i, self.n_i), dtype= torch.float), 'phase_weights': self.init_weights[f'layer_blocks.{self.n_layers}.phase_weights']}
            self.layer_blocks.append(d2nn_layer(n_hidden, self.n_o, self.out_dist, self.lambda_, neuron_size= self.neuron_size, learn_type=self.learn_type, device= self.device, energy_type = self.energy_type, window_size= self.window_size, RI = self.RI_Gel, weights= weight_dict, cfg = cfg))
        else:
            self.layer_blocks.append(d2nn_layer(n_hidden, self.n_o, self.out_dist, self.lambda_, neuron_size= self.neuron_size, learn_type=self.learn_type, device= self.device, energy_type = self.energy_type, window_size= self.window_size, RI = self.RI_Gel, cfg = cfg))
        
        if cfg['output_scalebias_matrix']:
            warnings.warn("Warning! You are attempting to use a Matrix to scale outputs !")
            
            if cfg['output_scale'] == 'randn':
                output_scale = torch.randn(1, self.n_o, self.n_o)
            else:
                output_scale = cfg['output_scale'] * torch.ones(1, self.n_o, self.n_o)
                
            if cfg['output_bias'] == 'randn':
                output_bias = torch.randn(1, self.n_o, self.n_o)
            else:
                output_bias = cfg['output_bias'] * torch.ones(1, self.n_o, self.n_o)
        else:
            output_scale=torch.tensor(cfg['output_scale'])
            output_bias= torch.tensor(cfg['output_bias'])
            
        if cfg['output_scale_learnable']:
            self.output_scale= nn.Parameter(output_scale)
        else:
            self.output_scale= output_scale
            
        if cfg['output_bias_learnable']:
            self.output_bias= nn.Parameter(output_bias)
        else:
            self.output_bias= output_bias

    def forward(self, input_e_field, epoch_n, batch_iter = None, cfg = None):
        '''
            Function for forward pass

            Args:
                input_e_field  : Input electric field (batch_size, self.n_i, self.n_i)
            Returns:
                output_e_field : Output E-field of D2NN
        '''
        x= input_e_field.view(-1, self.n_i*self.samples, self.n_i*self.samples)
        device = input_e_field.device
        
        mask_list = []   ### masks for each layer (INQ)
        for idx in range(len(self.layer_blocks)):
            x, temp, mask = self.layer_blocks[idx](x,epoch_n, batch_iter, cfg)
            if idx > 0:
                mask_list.append(mask)
        return x, self.output_bias.to(device), self.output_scale.to(device), temp, mask_list
