import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

def get_G_with_window(n_neurons_input = 300, neuron_size = 0.0003, delta_z = 0.004, lambda_ = 750e-6, w = 4,mask_factor = 1):
    global non_evancent_area
    '''
        Function to calculate the propagation transfer function (frequency domain)

        Args:
            n_neurons_input  : number of neurons on one side of the input layer
            neuron_size      : size of a neuron
            delta_z          : distance between two adjacent layers
            lambda_          : wavelength
            w                : window size
    
        Returns:
            G                : Propogation transfer function (frequency domain) 
    '''
    dx= neuron_size
    N= n_neurons_input
    
    # if (delta_z == 0): #usecase : delta_z =0 , there is no diffraction.
    #     return torch.ones((w*N,w*N), dtype=torch.cfloat)
    
    fx = torch.arange(-1/(2*dx),1/(2*dx),1/(w*N*dx)) 
    fx = torch.tile(fx, (1,w*N)).view(w*N,w*N).to(torch.cfloat)

    fy = torch.arange(1/(2*dx),-1/(2*dx),-1/(w*N*dx)).view(w*N,1)
    fy = torch.tile(fy, (1,w*N)).view(w*N,w*N).to(torch.cfloat)

    non_evancent_area = (abs(fx)**2 + abs(fy)**2 <= (1/lambda_)**2*mask_factor)

    power_for_G= 1j*2*np.pi*torch.sqrt((1/lambda_**2)-(fx**2)-(fy**2))*delta_z
    G= torch.exp(power_for_G*non_evancent_area)*non_evancent_area
    return G

class d2nnASwWindow_layer(nn.Module):
    '''
        A diffractive layer of the D2NN
        - uses Angular Spectrum Method to compute wave propagation
    '''
    def __init__(self, n_neurons_input, n_neurons_output, delta_z, lambda_, neuron_size, learn_type='both', device= 'cpu', window_size= 4, weights= None, mask_factor=1, **kwargs):
        '''
            Initlialize the diffractive layer

            Args : 
                n_neurons_input : Number of input neurons
                n_neurons_output : Number of output neurons
                delta_z : Distance between two adjacent layers of the D2NN
                lambda_ : Wavelength
                neuron_size : Size of the neuron
                learn_type : Type of learnable transmission coefficients
                device  : Device to run the model on
                window_size : Angular Spectrum method computational window size factor(default=4)
                weights : Weights to be loaded if using a pretrained model
        '''
        super(d2nnASwWindow_layer, self).__init__()
        self.n_i               = n_neurons_input
        self.n_o               = n_neurons_output
        self.delta_z           = delta_z
        self.lambda_           = lambda_
        self.neuron_size       = neuron_size
        self.learn_type        = learn_type
        self.w                 = window_size
        self.mask_factor       = mask_factor
        self.device            = device
        self.G = get_G_with_window(self.n_i, self.neuron_size, self.delta_z, self.lambda_, w= self.w,mask_factor=self.mask_factor).to(device) # Obtain frequency domain diffraction propogation function/ transformation

        if weights!= None:
            amp_weights= weights['amp_weights']
            phase_weights= weights['phase_weights']    
            
        
        if (self.learn_type=='amp'): # Learn only the amplitides of transmission coefficients in diffraction layers
            print('Learnable transmission coefficient: Amplitude only')
            if weights== None:
                self.amp_weights = nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
                self.phase_weights= torch.zeros((self.n_i, self.n_i), dtype= torch.float).to(device)
            else:
                print('loading weights ... ')
                self.amp_weights = nn.Parameter(amp_weights, requires_grad= True)
                self.phase_weights= phase_weights.to(device)   
        elif (self.learn_type=='phase'): # Learn only the phases of transmission coefficients in diffraction layers
            print('Learnable transmission coefficient: Phase only')
            if weights== None:
                self.amp_weights = torch.ones((self.n_i, self.n_i), dtype= torch.float).to(device) *100000
                self.phase_weights= nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
            else:
                print('loading weights ... ')
                self.amp_weights = amp_weights.to(device)
                self.phase_weights= nn.Parameter(phase_weights, requires_grad= True)
                
        elif (self.learn_type=='both'):  # Learn both the amplitides, phases of transmission coefficients in diffraction layers
            print('Learnable transmission coefficient: Amplitude and Phase')
            if weights== None:
                self.phase_weights= nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
                self.amp_weights = nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
            else:
                print('loading weights ... ')
                self.phase_weights= nn.Parameter(phase_weights, requires_grad= True)
                self.amp_weights = nn.Parameter(amp_weights, requires_grad= True)
        else: # Diffraction layers do not have learnable transmission coefficients
            print('No learnable transmission coefficients')
            if weights== None:
                self.phase_weights= torch.zeros((self.n_i, self.n_i), dtype= torch.float).to(device)
                self.amp_weights = torch.ones((self.n_i, self.n_i), dtype= torch.float).to(device)*100000
            else:
                print('loading weights ... ')
                self.phase_weights= phase_weights.to(device)
                self.amp_weights = amp_weights.to(device)*100000  
    
    def find_transfer_function(self, delta_z,mask_factor):
        self.G = get_G_with_window(self.n_i, self.neuron_size, delta_z, self.lambda_, w= self.w,mask_factor=mask_factor).to(self.device)
                
    def forward(self, input_e_field):
        '''
            Function for forward pass

            Args: 
                input_e_field  : Input electric field (batch_size, self.n_i, self.n_i) immediately before the current layer

            Returns:
                output_e_field : Output electric field (batch_size, self.n_o, self.n_o) immediately before the next layer
        '''
        device = input_e_field.device
        batch_size = input_e_field.shape[0]
        
        ts = (torch.sigmoid(self.amp_weights) * torch.exp(1j*self.phase_weights)).view(1, self.n_i, self.n_i) # Obtain transmission coefficients
        
        input_e_field = input_e_field.view(batch_size, self.n_i, self.n_i)
        
        im_pad= torch.zeros((batch_size, self.w*self.n_i, self.w*self.n_i), dtype= torch.cfloat).to(device)
        im_pad[:, (self.w-1)*self.n_i//2:self.n_i+(self.w-1)*self.n_i//2, (self.w-1)*self.n_i//2:self.n_i+(self.w-1)*self.n_i//2]= input_e_field #* ts # field is propogated through the layer which can be larger than the field (i.e. determine by self.w)

        A= torch.fft.fftshift(torch.fft.fft2(im_pad)) * self.G.view(1, self.w*self.n_i, self.w*self.n_i) # propogation (from just after the current layer to the next layer) in frequency domain
        B= torch.fft.ifft2(torch.fft.ifftshift(A)) # convert back to spatial domain
        
        U = B[:, (self.w-1)*self.n_i//2:(self.w-1)*self.n_i//2+self.n_i, (self.w-1)*self.n_i//2:(self.w-1)*self.n_i//2+self.n_i]
        
        output_e_field = U

        return output_e_field