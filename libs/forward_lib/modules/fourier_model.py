import torch
import numpy as np
from torch import nn
from modules.diffraction import *
from modules.d2nn_layers import *
from modules.quantization import *


class d2nn_fourier(nn.Module):
    '''
        Diffractive Deep Neural Network implemented in the Fourier domain
    '''
    def __init__(self, cfg, d2nn_layer= d2nnASwWindow_layer, weights= None):
        super(d2nn_fourier, self).__init__()
        
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
        self.in_dist = cfg['in_dist']
        self.out_dist= cfg['out_dist']
        self.window_size= cfg['window_size']
        
        if weights== None:
            weights= {}
            for idx in range(self.n_layers):
                weights[f'layer{idx}']= None
        
        n_hidden= (self.n_i+ self.n_o)//2
        #d2nn_layer= d2nn_layer
  
        self.layer_blocks: nn.ModuleList[d2nn_layer] = nn.ModuleList()
        #self.layer_blocks.append(d2nn_layer(self.n_i, n_hidden, self.delta_z, self.lambda_, neuron_size= self.neuron_size, learn_type='no', device= self.device))
        self.layer_blocks.append(d2nn_layer(self.n_i, n_hidden, self.in_dist, self.lambda_, neuron_size= self.neuron_size, learn_type='no', device= self.device, window_size= self.window_size))

        for idx in range(self.n_layers-1):
            self.layer_blocks.append(d2nn_layer(n_hidden, n_hidden, self.delta_z, self.lambda_, neuron_size= self.neuron_size, learn_type=self.learn_type, device= self.device, energy_type = self.energy_type, window_size= self.window_size, weights= weights[f'layer{idx}']))
            
        self.layer_blocks.append(d2nn_layer(n_hidden, self.n_o, self.out_dist, self.lambda_, neuron_size= self.neuron_size, learn_type=self.learn_type, device= self.device, energy_type = self.energy_type, window_size= self.window_size, weights= weights[f'layer{self.n_layers-1}']))
        
    def forward(self, input_e_field):
        '''
            Function for forward pass
            Args:
                input_e_field  : Input electric field (batch_size, self.n_i, self.n_i)
            Returns:
                output_e_field : Output E-field of D2NN
        '''
        x= input_e_field.view(-1, self.n_i, self.n_i)
        device = input_e_field.device

        Fs = torch.fft.fft2(x)
        X = torch.fft.fftshift(Fs)
            
        for idx in range(len(self.layer_blocks)):
            X= self.layer_blocks[idx](X)
        
        x_o = torch.fft.ifft2(torch.fft.ifftshift(X))

        return x_o

class fourier_layer(nn.Module):
    '''
        Learnable Fourier Filter in the 4-F system
    '''
    def __init__(self, n_neurons_input, n_neurons_output, neuron_size, learn_type='both', device= 'cpu', weights= None, circular = False, **kwargs):
        '''
            Initialize the Fourier Filter

            Args : 
                n_neurons_input  : number of neurons in the input layer
                n_neurons_output : number of neurons in the output layer
                neuron_size : size of the neurons in the input layer
                learn_type  : type of learning to be used for the filter ('amp','phase','both')
                device  : device to be used for the filter
                weights : weights to be used for the filter (if pretrained filter is used)
                circular: whether the filter is circular or not
        '''

        super(fourier_layer, self).__init__()
        self.n_i               = n_neurons_input
        self.n_o               = n_neurons_output
        self.neuron_size       = neuron_size
        self.learn_type        = learn_type
        self.circular          = circular # if the filter is circular or not
        self.cfg               = kwargs['cfg']

        self.fpr                = self.cfg['full_precision_and_range']
        self.hard_quant         = self.cfg['hard_quant']
        self.fake_quant         = self.cfg['fake_quant']

        self.quant_levels       = self.cfg['quant_levels']
        self.lower_bound        = self.cfg['lower_bound']
        self.upper_bounds       = self.cfg['upper_bounds']

        self.quant_after        = self.cfg['quant_after']
        self.schedule_start     = self.cfg['schedule_start']
        self.schedule_array     = self.cfg['schedule_array']
        self.schedule_every     = self.cfg['schedule_every']
        self.dsq_factor         = self.cfg['dsq_factor']

        self.quant_func         = kwargs['quant_func'] if 'quant_func' in kwargs.keys() else None
        
        if weights!= None:
            amp_weights= weights['amp_weights']
            phase_weights= weights['phase_weights']    
            
        
        if (self.learn_type=='amp'):
            print('Learnable transmission coefficient: Amplitude only')
            if weights== None:
                self.amp_weights = nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
                self.phase_weights= torch.zeros((self.n_i, self.n_i), dtype= torch.float).to(device)
            else:
                print('loading weights ... ')
                self.amp_weights = nn.Parameter(amp_weights, requires_grad= True)
                self.phase_weights= phase_weights.to(device)   
        elif (self.learn_type=='phase'):
            print('Learnable transmission coefficient: Phase only')
            if weights== None:
                self.amp_weights = torch.ones((self.n_i, self.n_i), dtype= torch.float).to(device) *100000
                self.phase_weights= nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
            else:
                print('loading weights ... ')
                self.amp_weights = amp_weights.to(device)
                self.phase_weights= nn.Parameter(phase_weights, requires_grad= True)
                
        elif (self.learn_type=='both'):
            print('Learnable transmission coefficient: Amplitude and Phase')
            if weights== None:
                self.phase_weights= nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
                self.amp_weights = nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
            else:
                print('loading weights ... ')
                self.phase_weights= nn.Parameter(phase_weights, requires_grad= True)
                self.amp_weights = nn.Parameter(amp_weights, requires_grad= True)
        else:
            print('No learnable transmission coefficients')
            if weights== None:
                self.phase_weights= torch.zeros((self.n_i, self.n_i), dtype= torch.float).to(device)
                self.amp_weights = torch.ones((self.n_i, self.n_i), dtype= torch.float).to(device)*100000
            else:
                print('loading weights ... ')
                self.phase_weights= phase_weights.to(device)
                self.amp_weights = amp_weights.to(device)*100000  

    def scheduler(self, epoch):
        if self.cfg['learn_schedule']:
            return 1/(torch.abs(self.quant_func.k)+0.5e-2)
        elif epoch <= self.quant_after:
            return self.schedule_start
        else:
            return self.schedule_array[((epoch-self.quant_after)//self.schedule_every)]
        
    def dsq_scheduler(self, epoch):
        if self.cfg['dsq_temp_learn']:
            # dsq_reg_l = self.cfg['dsq_regularize'][0]
            # dsq_reg_u = self.cfg['dsq_regularize'][1]
            # slope = dsq_reg_l + (dsq_reg_u - dsq_reg_l) * torch.sigmoid(self.quant_func.alpha)
            slope = torch.abs(self.quant_func.alpha) + 1e-8
        elif self.cfg['dsq_temp_const']:
            slope = self.cfg['dsq_alpha']
        else:
            slope = np.exp(-1*(self.scheduler(epoch)+1)/self.dsq_factor)
        
        return slope
    
    def __convert_raw_weights(self, epoch_n = None, device = None):
        if self.cfg['dsq']:
            slope_factor = self.dsq_scheduler(epoch_n)
        else:
            slope_factor = self.scheduler(epoch_n)

        if(device == None):
            device = self.cfg['device']

        if self.learn_type == 'no':
            scaled_phase_weights = self.phase_weights
        elif self.fpr:
            scaled_phase_weights = self.phase_weights
        elif self.hard_quant:
            scaled_phase_weights = self.quant_func(self.phase_weights, 100, train = False)
            
        ##### Fake Quantization
        elif self.fake_quant:
            scaled_phase_weights = fake_quant.apply(self.phase_weights, self.quant_levels[0], self.lower_bound, self.upper_bounds[0])
        
        ##### Differentiable Soft Quantization
        elif self.cfg['dsq']:
            if ((self.training) and (epoch_n <= self.quant_after)):
                scaled_phase_weights = self.phase_weights
            elif ((self.training) and (epoch_n > self.quant_after)):
                scaled_phase_weights = self.quant_func(self.phase_weights, k=slope_factor, train=True, noise = 0)
            else:
                scaled_phase_weights = self.quant_func(self.phase_weights, k=slope_factor, train=False, noise = 0)
        
        ##### Quantization using multi level sigmoid
        else:
            if ((self.training) and (epoch_n <= self.quant_after)):
                scaled_phase_weights = self.phase_weights
            elif ((self.training) and (epoch_n > self.quant_after)):
                scaled_phase_weights = self.quant_func(self.phase_weights, k=slope_factor, train=True, noise = 0)
            else:
                scaled_phase_weights = self.quant_func(self.phase_weights, k=100, train=False, noise = 0)
            
        return scaled_phase_weights, slope_factor
                
    def forward(self, input_e_field, epoch_n = None):
        '''
            Forward pass of the Fourier Filter

            Args:
                input_e_field : input electric field (batch_size, self.n_i, self.n_i)
                epoch_n       : current epoch number

            Returns:    
                output_e_field : output electric field
        '''
        device = input_e_field.device
        batch_size = input_e_field.shape[0]

        scaled_phase_weights, slope_factor = self.__convert_raw_weights(epoch_n, device)
        
        ts = (torch.sigmoid(self.amp_weights) * torch.exp(1j*scaled_phase_weights)).view(1, self.n_i, self.n_i)
        if self.circular:
            rc = self.n_i//2
            xc = torch.arange(-self.n_i//2,self.n_i//2,1) 
            xc = torch.tile(xc, (1,self.n_i)).view(self.n_i,self.n_i).to(torch.cfloat)

            yc = torch.arange(self.n_i//2,-self.n_i//2,-1).view(self.n_i,1)
            yc = torch.tile(yc, (1,self.n_i)).view(self.n_i,self.n_i).to(torch.cfloat)

            circ = (abs(xc)**2 + abs(yc)**2 <= (rc)**2).to(torch.float32).view(1,self.n_i,self.n_i).to(device)
            
            ts = ts * circ
        
        input_e_field = input_e_field.view(batch_size, self.n_i, self.n_i)
        
        output_e_field = input_e_field * ts

        return output_e_field, slope_factor

class fourier_model(nn.Module):
    '''
        Learnable Fourier Filter Model Class
    '''
    def __init__(self, cfg, weights=None, layer= fourier_layer):
        '''
            Initialize the Fourier Filter Model

            Args : 
                cfg : configuration dictionary
                layer : layer class to be used for the filter
        '''
        super(fourier_model, self).__init__()
        
        self.n_i = cfg['img_size']
        self.n_o= cfg['img_size']
        self.neuron_size= cfg['neuron_size']
        self.learn_type= cfg['learn_type']
        self.device= cfg['device']
        self.n_layers= 1
        self.circular = cfg['filter_circular'] if 'filter_circular' in cfg.keys() else False
        self.weights = weights
        
        if self.weights== None:
            self.weights= {}
            for idx in range(self.n_layers):
                self.weights[f'layer{idx}']= None

        if cfg['dsq']:
            self.quant_function = dsq(cfg['quant_levels'], cfg['lower_bound'], cfg['upper_bounds'], cfg['learn_u'], cfg['dsq_alpha'])
        else:
            self.quant_function = mlsq(cfg['quant_levels'], cfg['lower_bound'], cfg['upper_bounds'], cfg['learn_u'], cfg['alpha'])
  
        self.layer_blocks: nn.ModuleList[layer] = nn.ModuleList()
            
        self.layer_blocks.append(layer(self.n_i, self.n_o, neuron_size= self.neuron_size, learn_type=self.learn_type, device= self.device, weights= self.weights[f'layer{self.n_layers-1}'], circular = self.circular, cfg=cfg, quant_func=self.quant_function))
        
        output_scale= torch.tensor(cfg['output_scale'])
        output_bias = torch.tensor(cfg['output_bias'])
        
        if cfg['output_scale_learnable']:
            self.output_scale= nn.Parameter(torch.tensor(output_scale))
        else:
            self.output_scale= output_scale

        self.ouptut_bias  = output_bias
        
    def forward(self, input_e_field, epoch_n, batch_iter=None, cfg=None):
        '''
            Function for forward pass
            Args:
                input_e_field  : Input electric field (batch_size, self.n_i, self.n_i)
                epoch_n        : Current epoch
            Returns:
                output_e_field : Output E-field of the LFF
                output_scale   : Scaling factor for the reconstructed image
        '''
        x= input_e_field.view(-1, self.n_i, self.n_i)
        device = input_e_field.device

        Fs = torch.fft.fft2(x)
        X = torch.fft.fftshift(Fs)
            
        for idx in range(len(self.layer_blocks)):
            X, slope_factor = self.layer_blocks[idx](X, epoch_n)
        
        x_o = torch.fft.ifft2(torch.fft.ifftshift(X))

        return x_o, self.ouptut_bias.to(device), self.output_scale.to(device), slope_factor, None