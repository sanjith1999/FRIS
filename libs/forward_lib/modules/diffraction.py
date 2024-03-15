import torch
import numpy as np

def rayleigh_sommerfeld_diffraction(n_neurons_input, n_neurons_output, lambda_, delta_z, neuron_size):
    '''
        Function to calculate the rayleigh sommerfeld diffraction formula 
        
        Args:
            n_neurons_input  : number of neurons on one side of the input layer
            n_neurons_output : number of neurons on one side at the output
            lambda_          : wavelength
            delta_z          : distance between two adjacent layers
            neuron_size      : size of a neuron
        Returns:
            w_ip             : W matrix (n_neurons_input,n_neurons_input,n_neurons_output,n_neurons_output)
            r                : Radial distance (n_neurons_input,n_neurons_input,n_neurons_output,n_neurons_output)
    '''
    
    n_i = n_neurons_input
    n_p = n_neurons_output

    i_range = range(-n_i//2, n_i//2) # input coordinates
    p_range = range(-n_p//2, n_p//2) # output (next layer) coordinates

    xi = yi = torch.tensor(i_range).reshape(n_i,1)
    xp = yp = torch.tensor(p_range).reshape(1, n_p)
    
    delta_x = torch.tile(xi, (1,n_p)) - torch.tile(xp, (n_i,1))
    delta_y = torch.tile(yi, (1,n_p)) - torch.tile(yp, (n_i,1))

    delta_x = torch.tile(delta_x, (n_i, n_p, 1,1 )).permute(2,3,0,1) *neuron_size
    delta_y = torch.tile(delta_y, (n_i, n_p, 1,1 )) *neuron_size
    r = torch.sqrt(delta_x**2 + delta_y**2 + delta_z**2).permute(0,2,1,3)

    w_ip = ((delta_z/(r**2)) * (1/(2*np.pi*r) + 1/(1j * lambda_))* torch.exp(1j* 2*np.pi * r/lambda_))*(neuron_size**2)

    return w_ip, r

def get_G(n_neurons_input, neuron_size, delta_z, lambda_, device='cpu'):
    '''
        Function to calculate G (the fourier transform of g(x,y,z))
        
    '''
    
    layer_s_len = n_neurons_input * neuron_size # the length of a side of the layer in m

    range_ = torch.arange(-n_neurons_input//2,n_neurons_input//2).to(torch.complex64).to(device)
    fx = torch.tile(range_.reshape(1, n_neurons_input), (n_neurons_input,1)) * 1/layer_s_len 
    fy = torch.tile(range_.reshape(n_neurons_input,1), (1,n_neurons_input)) * 1/layer_s_len
    fx2 = fx**2
    fy2 = fy**2

    mask = ((abs(fx2) + abs(fy2)) <= (1/(lambda_**2))).to(torch.float32)

    return mask*torch.exp(1j * 2*np.pi * torch.sqrt(mask*((1/lambda_**2) - (fx2 + fy2))) * delta_z).to(device) 


def get_G_with_window(n_neurons_input= 300, neuron_size= 0.0003, delta_z= 0.004, lambda_air = 750e-6, w= 2, RI = 1):
    dx = neuron_size
    N  = n_neurons_input

    lambda_ = lambda_air / RI
    
    if (delta_z == 0): #usecase : delta_z =0 , there is no diffraction.
        return torch.ones((w*N,w*N), dtype=torch.cfloat)
    
    # print("dx = ", dx, "N = ", N, "w = ", w, "lambda_ = ", lambda_, "delta_z = ", delta_z, "RI = ", RI)
    # print("1/(2*dx) ", 1/(2*dx), "1/(w*N*dx) ", 1/(w*N*dx))
    
    # fx = torch.arange(-1/(2*dx),1/(2*dx),1/(w*N*dx)) 
    fx = torch.linspace(-1/(2*dx),1/(2*dx),w*N)
    fx = torch.tile(fx, (1,w*N)).view(w*N,w*N).to(torch.cfloat)


    # fy = torch.arange(1/(2*dx),-1/(2*dx),-1/(w*N*dx)).view(w*N,1)
    fy = torch.linspace(1/(2*dx), -1/(2*dx), w*N).view(w*N,1)
    fy = torch.tile(fy, (1,w*N)).view(w*N,w*N).to(torch.cfloat)

    non_evancent_area = (abs(fx)**2 + abs(fy)**2 <= (1/lambda_)**2)

    power_for_G= 1j*2*np.pi*torch.sqrt((1/lambda_**2)-(fx**2)-(fy**2))*delta_z
    G= torch.exp(power_for_G*non_evancent_area)*non_evancent_area
    
    return G