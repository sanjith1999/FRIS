import torch
import torch.nn as nn
import numpy as np
import scipy.io
from scipy.optimize import curve_fit

from libs.forward_lib.modules.diffraction import *

def poly_fit(x, n, a, device = 'cpu'):
    '''
        generates the nth order polynomial y = f(x).
        
        args:
            x: independent parameter
            n: order of the polynomial
            a: coefficients of the polynomial with the coefficient of the highest order first
        returns:
            y = f(x) where f(x) is the nth order polynomial with coefficients a
    '''
    x = x.unsqueeze(-1).to(device)
    N = torch.arange(n,-1,-1).to(device)
    y = x**N
    z = y*a
    return torch.sum(z, dim=-1)

def gs_fit(x, c, d):
    return (1 / (1 + np.exp(-c * (x + d))))

def gs_func(x, c, d):
    return (1 / ( 1 + torch.exp(-c * (x + d))))

def calc_error(y, y_hat):
    '''
        calculates the sum of squared error.
        
        args:
            y    : ground truth
            y_hat: predicted values
        returns:
            sum of squared errors of the prediction
    '''
    return torch.sum((y-y_hat)**2)

def fit_gs_population(i0, gs, n=15):
    '''
        fits a polynomial of order n to the ground state population.
        
        args:
            i0 : input intensity range (usually 1e9)
            gs : ground state population data
            n  : order of the polynomial
        returns:
            p  : coefficients of the fitted polynomial; gs = f(log(i0)) where f is a polynomial of nth order
            M0 : initial concentration
    '''
    # p = np.polyfit(torch.log10(i0)[0], gs[0], n)
    p, pcov = curve_fit(gs_fit, torch.log10(i0)[0].numpy(), gs[0].numpy(), sigma=np.ones(i0[0].shape)*0.00001, method='lm')
    p = torch.from_numpy(p)
    # gs_hat = poly_fit(torch.log10(i0)[0], n, p)
    gs_hat = gs_func(torch.log10(i0)[0], p[0], p[1])
    M0 = gs_hat[0]
    print(f"Sigmoid function fitted (SSE: {calc_error(gs,gs_hat.T):.6f}).")
    return p, M0

class sat_abs_layer(nn.Module):
    '''
        creates a saturable absorption layer (optical non-linear layer).
        
        args:
            init_abs       : initial absorption (keep in the range [1,5])
            n              : order of the polynomial to fit the ground state population
            device         : device for computation ('cpu' or 'gpu')
            max_intensity  : maximum intensity of the field (keep within 1e7 to 1e8 for better non-linearity)
            data_path      : path to the folder where ground state population data stored
    '''
    
    def __init__(self, n_neurons_input, n_neurons_output, delta_z, lambda_, neuron_size, init_abs, window_size= 2, RI = 1, n = 15, device = 'cpu', cfg = None, max_intensity = 5e6, data_path = "saturable_absorption/data", test_mode = False):
        
        super(sat_abs_layer, self).__init__()
        
        self.n_i         = n_neurons_input
        self.n_o         = n_neurons_output
        self.delta_z     = delta_z
        self.lambda_     = lambda_
        self.neuron_size = neuron_size
        self.init_abs    = init_abs
        self.w           = window_size
        self.RI_medium   = RI
        self.n           = n
        self.max_int     = max_intensity
        self.device      = device
        self.cfg         = cfg
        self.samples     = cfg['samples'] if cfg!=None else 1
        self.test_mode   = test_mode
        
        gs = torch.from_numpy(scipy.io.loadmat(f"{data_path}/gs_pop_108.mat")['gs'])
        i0 = torch.from_numpy(scipy.io.loadmat(f"{data_path}/input_flux_108.mat")['re'])
        self.p, self.M0 = fit_gs_population(i0, gs, self.n)
        self.p = self.p.to(device)
        self.sigZ = (self.init_abs/self.M0).to(device)
        
        self.G = get_G_with_window(self.n_i*self.samples, self.neuron_size/self.samples, self.delta_z, self.lambda_, w= self.w, RI = self.RI_medium).to(device)
        
    def forward(self, input_e_field, epoch_n=None, batch_iter=None, cfg=None):
        '''
            function for the forward pass
            
            args:
                input_e_field  : input electric field to the saturable absorption layer
            returns:
                output_e_field : output electric field from the saturable absorption layer
        '''
        batch_size = input_e_field.shape[0]
        
        input_int = ((torch.abs(input_e_field)**2) + (1e4/self.max_int)) * self.max_int
        # gs_pop = poly_fit(torch.log10(input_int), self.n, self.p, self.device)
        gs_pop = gs_func(torch.log10(input_int), self.p[0], self.p[1])
        output_int = input_int * torch.exp(-1*self.sigZ*gs_pop) / self.max_int
        transformed_e_field = torch.sqrt(output_int) * torch.exp(1j*torch.angle(input_e_field))
        
        transformed_e_field = transformed_e_field.view(batch_size, self.n_i*self.samples, self.n_i*self.samples)
        
        im_pad= torch.zeros((batch_size, self.w*self.n_i*self.samples, self.w*self.n_i*self.samples), dtype= torch.cfloat).to(self.device)
        im_pad[:, (self.w-1)*self.n_i*self.samples//2:self.n_i*self.samples+(self.w-1)*self.n_i*self.samples//2, (self.w-1)*self.n_i*self.samples//2:self.n_i*self.samples+(self.w-1)*self.n_i*self.samples//2]= transformed_e_field

        A= torch.fft.fftshift(torch.fft.fft2(im_pad)) * self.G.view(1, self.w*self.n_i*self.samples, self.w*self.n_i*self.samples)
        B= torch.fft.ifft2(torch.fft.ifftshift(A))
        
        U = B[:, (self.w-1)*self.n_i*self.samples//2:(self.w-1)*self.n_i*self.samples//2+self.n_i*self.samples, (self.w-1)*self.n_i*self.samples//2:(self.w-1)*self.n_i*self.samples//2+self.n_i*self.samples]
        
        output_e_field = U
        
        if self.test_mode:
            return output_e_field, transformed_e_field, gs_pop
        else:
            return output_e_field, 0, 0