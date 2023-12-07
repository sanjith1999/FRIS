import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

def get_G_with_window(Nx = 300, Ny = 300, dx = 3e-4, dy = 3e-4, dz = 4e-3, lambda_ = 750e-6, w = 4,mask_factor = 1):
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
    
    # if (delta_z == 0): #usecase : delta_z =0 , there is no diffraction.
    #     return torch.ones((w*N,w*N), dtype=torch.cfloat)
    
    fx = torch.arange(-1/(2*dx),1/(2*dx),1/(w*Nx*dx)) 
    fx = torch.tile(fx, (1,w*Nx)).view(w*Nx,w*Ny).to(torch.cfloat)

    fy = torch.arange(1/(2*dy),-1/(2*dy),-1/(w*Ny*dy)).view(w*Ny,1)
    fy = torch.tile(fy, (1,w*Ny)).view(w*Nx,w*Ny).to(torch.cfloat)

    non_evancent_area = (abs(fx)**2 + abs(fy)**2 <= (1/lambda_)**2*mask_factor)

    power_for_G= 1j*2*np.pi*torch.sqrt((1/lambda_**2)-(fx**2)-(fy**2))*dz
    G= torch.exp(power_for_G*non_evancent_area)*non_evancent_area
    return G

class d2nnASwWindow_layer(nn.Module):
    '''
        A diffractive layer of the D2NN
        - uses Angular Spectrum Method to compute wave propagation
    '''
    def __init__(self, Nx, Ny, dx, dy, dz, lambda_, device= 'cpu', window_size= 4, mask_factor=1, **kwargs):
        '''
            Initlialize the diffractive layer
        '''
        super(d2nnASwWindow_layer, self).__init__()
        self.Nx, self.Ny       = Nx, Ny
        self.dx, self.dy       = dx, dy
        self.dz                = dz
        self.lambda_           = lambda_
        self.w                 = window_size
        self.mask_factor       = mask_factor
        self.device            = device
        self.G = get_G_with_window().to(device) # Obtain frequency domain diffraction propogation function/ transformation
            
    
    def find_transfer_function(self,dz_, mask_factor_):
        self.G = get_G_with_window(Nx = self.Nx, Ny = self.Ny, dx = self.dx, dy = self.dy, dz = dz_, lambda_ = self.lambda_, w = self.w,mask_factor = mask_factor_).to(self.device)
                
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
        
        
        input_e_field = input_e_field.view(batch_size, self.Nx, self.Ny)
        
        im_pad= torch.zeros((batch_size, self.w*self.Nx, self.w*self.Ny), dtype= torch.cfloat).to(device)
        im_pad[:, (self.w-1)*self.Nx//2:self.Nx+(self.w-1)*self.Nx//2, (self.w-1)*self.Ny//2:self.Ny+(self.w-1)*self.Ny//2]= input_e_field #* ts # field is propogated through the layer which can be larger than the field (i.e. determine by self.w)

        A= torch.fft.fftshift(torch.fft.fft2(im_pad)) * self.G.view(1, self.w*self.Nx, self.w*self.Ny) # propogation (from just after the current layer to the next layer) in frequency domain
        B= torch.fft.ifft2(torch.fft.ifftshift(A)) # convert back to spatial domain
        
        U = B[:, (self.w-1)*self.Nx//2:(self.w-1)*self.Nx//2+self.Nx, (self.w-1)*self.Ny//2:(self.w-1)*self.Ny//2+self.Ny]
        
        output_e_field = U

        return output_e_field