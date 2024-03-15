import torch
import numpy as np
from torch import nn
from modules.diffraction import *

class d2nn_layer(nn.Module):
    '''
        A diffractive layer of the D2NN
    '''
    def __init__(self, n_neurons_input, n_neurons_output, delta_z, lambda_, neuron_size, learn_type='both', device= 'cpu', energy_type= 'passive'):
        super(d2nn_layer, self).__init__()
        self.n_i               = n_neurons_input
        self.n_o               = n_neurons_output
        self.delta_z           = delta_z
        self.lambda_           = lambda_
        self.neuron_size       = neuron_size
        self.learn_type        = learn_type
        self.energy_type       = energy_type

        self.w_ip = rayleigh_sommerfeld_diffraction(self.n_i, self.n_o, self.lambda_, self.delta_z, self.neuron_size)[0].view(1, self.n_i, self.n_i, self.n_o, self.n_o).to(device)

        if (self.learn_type=='amp'):
            print('Learnable transmission coefficient: Amplitude only')
            self.amp_weights = nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
            self.phase_weights= torch.zeros((self.n_i, self.n_i), dtype= torch.float).to(device)
        elif (self.learn_type=='phase'):
            print('Learnable transmission coefficient: Phase only')
            self.amp_weights = torch.ones((self.n_i, self.n_i), dtype= torch.float).to(device)*100000
            self.phase_weights= nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
        elif (self.learn_type=='both'):
            print('Learnable transmission coefficient: Amplitude and Phase')
            self.phase_weights= nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
            self.amp_weights = nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
        else:
            print('No learnable transmission coefficients')
            self.phase_weights= torch.zeros((self.n_i, self.n_i), dtype= torch.float).to(device)
            self.amp_weights = torch.ones((self.n_i, self.n_i), dtype= torch.float).to(device)*100000
        
    def forward(self, input_e_field, take_sum=True):
        '''
            Function for forward pass
            
            Args:
                input_e_field  : Input electric field (batch_size, self.n_i, self.n_i)
                take_sum       : Condition to return the summed output E-field or E-field (for visualization purposes)
            Returns:
                output_e_field : Summed output E-field or E-field (for visualization purposes)
        '''
        device = input_e_field.device
        batch_size = input_e_field.shape[0]
        input_e_field = input_e_field.view(batch_size, self.n_i, self.n_i, 1, 1)
        
        if self.energy_type== 'passive':
            ts = (torch.sigmoid(self.amp_weights) * torch.exp(1j*self.phase_weights)).view(1, self.n_i, self.n_i, 1, 1) 
        else:
            ts = (torch.abs(self.amp_weights) * torch.exp(1j*self.phase_weights)).view(1, self.n_i, self.n_i, 1, 1)
        output_e_field = self.w_ip*ts*input_e_field

        if take_sum:
            output_e_field = torch.sum(output_e_field, dim=[1, 2])
        #print(f'output_e_field.range : {output_e_field.abs().min()}, {output_e_field.abs().max()}')
        return output_e_field

class d2nn(nn.Module):
    '''
        Diffractive Deep Neural Network
    '''
    def __init__(self, cfg):
        super(d2nn, self).__init__()
        
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
        
        n_hidden= (self.n_i+ self.n_o)//2
  
        self.layer_blocks: nn.ModuleList[d2nn_layer] = nn.ModuleList()
        self.layer_blocks.append(d2nn_layer(self.n_i, n_hidden, self.in_dist, self.lambda_, neuron_size= self.neuron_size, learn_type='no', device= self.device))
        
        for idx in range(self.n_layers):
            self.layer_blocks.append(d2nn_layer(n_hidden, n_hidden, self.delta_z, self.lambda_, neuron_size= self.neuron_size, learn_type=self.learn_type, device= self.device, energy_type = self.energy_type))
            
        self.layer_blocks.append(d2nn_layer(n_hidden, self.n_o, self.out_dist, self.lambda_, neuron_size= self.neuron_size, learn_type='no', device= self.device, energy_type = self.energy_type))

        
        self.ramp = torch.tile(torch.tensor(range(self.n_i))/self.n_i*np.pi*25, (self.n_i, 1)).to(self.device)
        self.ejdelta = torch.exp(1j*self.ramp)
        
    def forward(self, input_e_field):
        '''
            Function for forward pass

            Args:
                input_e_field  : Input electric field (batch_size, self.n_i, self.n_i)
            Returns:
                output_e_field : Output E-field of D2NN
        '''
        x= input_e_field.view(-1, self.n_i, self.n_i)

        if self.int_ramp == 'begin':
            x  = x + self.ejdelta
            
        for idx in range(len(self.layer_blocks)):
            x= self.layer_blocks[idx](x)

        if self.int_ramp == 'end':
            x  = x + self.ejdelta
        
        return x

##################################################################################################################################

class d2nnAS_layer(nn.Module):
    def __init__(self, n_neurons_input, n_neurons_output, delta_z, lambda_, neuron_size, learn_type='both', device= 'cpu'):
        super(d2nnAS_layer, self).__init__()
        self.n_i               = n_neurons_input
        self.n_o               = n_neurons_output
        self.delta_z           = delta_z
        self.lambda_           = lambda_
        self.neuron_size       = neuron_size
        self.learn_type        = learn_type
        
        self.G = get_G(self.n_i, self.neuron_size, self.delta_z, self.lambda_, device=device)

        if (self.learn_type=='amp'):
            print('Learnable transmission coefficient: Amplitude only')
            self.amp_weights = nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
            self.phase_weights= torch.zeros((self.n_i, self.n_i), dtype= torch.float).to(device)
        elif (self.learn_type=='phase'):
            print('Learnable transmission coefficient: Phase only')
            self.amp_weights = torch.ones((self.n_i, self.n_i), dtype= torch.float).to(device)*100000
            self.phase_weights= nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
        elif (self.learn_type=='both'):
            print('Learnable transmission coefficient: Amplitude and Phase')
            self.phase_weights= nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
            self.amp_weights = nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
        else:
            print('No learnable transmission coefficients')
            self.phase_weights= torch.zeros((self.n_i, self.n_i), dtype= torch.float).to(device)
            self.amp_weights = torch.ones((self.n_i, self.n_i), dtype= torch.float).to(device)*100000

    def forward(self, input_e_field):
        device = input_e_field.device
        batch_size = input_e_field.shape[0]
        input_e_field = input_e_field.view(batch_size, self.n_i, self.n_i)

        AS = torch.fft.fft2(input_e_field)
        AS_shifted = torch.fft.fftshift(AS)
        U = torch.fft.ifft2(torch.fft.ifftshift(AS_shifted * self.G))

        ts = (torch.sigmoid(self.amp_weights) * torch.exp(1j*self.phase_weights)).view(1, self.n_i, self.n_i)   
        output_e_field = ts*U

        # for the last layer you have to multiply by a G since this will be travelling to the detector **************
        #if self.output:
        #    G_out = get_G(self.n_i, self.neuron_size, self.delta_z, self.lambda_, device=device)
        #    output_AS = torch.fft.fft2(output_e_field)
        #    output_AS_shifted = torch.fft.fftshift(output_AS)
        #    output_e_field = torch.fft.ifft2(torch.fft.ifftshift(output_AS_shifted * G_out))

        return output_e_field
    
class d2nn_AS(nn.Module):
    '''
        Diffractive Deep Neural Network implemented using the AS method
    '''
    def __init__(self, cfg):
        super(d2nn_AS, self).__init__()
        
        self.n_i = cfg['img_size']
        self.n_o= cfg['img_size']

        self.delta_z = cfg['delta_z']
        self.lambda_ = cfg['lambda_']
        self.int_ramp= cfg['int_ramp']
        self.neuron_size= cfg['neuron_size']
        self.learn_type= cfg['learn_type']
        self.device= cfg['device']
        self.n_layers= cfg['n_layers']
        self.in_dist = cfg['in_dist']
        self.out_dist = cfg['out_dist']
        
        n_hidden= (self.n_i+ self.n_o)//2
  
        self.layer_blocks: nn.ModuleList[d2nnAS_layer] = nn.ModuleList()
        
        self.layer_blocks.append(d2nnAS_layer(n_hidden, n_hidden, self.in_dist, self.lambda_, neuron_size= self.neuron_size, learn_type=self.learn_type, device= self.device))
        
        for idx in range(self.n_layers-1):
            self.layer_blocks.append(d2nnAS_layer(n_hidden, n_hidden, self.delta_z, self.lambda_, neuron_size= self.neuron_size, learn_type=self.learn_type, device= self.device))
            
        self.layer_blocks.append(d2nnAS_layer(n_hidden, self.n_o, self.out_dist, self.lambda_, neuron_size= self.neuron_size, learn_type='no', device= self.device))
        
        self.ramp = torch.tile(torch.tensor(range(self.n_i))/self.n_i*np.pi*25, (self.n_i, 1)).to(self.device)
        self.ejdelta = torch.exp(1j*self.ramp)
        
    def forward(self, input_e_field):
        '''
            Function for forward pass
            Args:
                input_e_field  : Input electric field (batch_size, self.n_i, self.n_i)
            Returns:
                output_e_field : Output E-field of D2NN
        '''
        x= input_e_field.view(-1, self.n_i, self.n_i)

        if self.int_ramp == 'begin':
            x  = x + self.ejdelta
            
        for idx in range(len(self.layer_blocks)):
            x= self.layer_blocks[idx](x)

        if self.int_ramp == 'end':
            x  = x + self.ejdelta
        
        return x
