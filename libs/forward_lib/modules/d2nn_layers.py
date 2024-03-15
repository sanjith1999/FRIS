import torch
import numpy as np
from torch import nn
from libs.forward_lib.modules.diffraction import *
from libs.forward_lib.modules.quantization import *
    

class d2nnASwWindow_layer(nn.Module):
    def __init__(self, n_neurons_input, n_neurons_output, delta_z, lambda_, neuron_size, learn_type='both', device= 'cpu', window_size= 2, weights= None, RI = 1, **kwargs):
        super(d2nnASwWindow_layer, self).__init__()
        self.n_i               = n_neurons_input
        self.n_o               = n_neurons_output
        self.delta_z           = delta_z
        self.lambda_           = lambda_
        self.neuron_size       = neuron_size
        self.learn_type        = learn_type
        self.w                 = window_size
        self.quant_after    = kwargs['cfg']['quant_after']          # start quantization after this many epochs
        self.schedule_start = kwargs['cfg']['schedule_start']
        self.schedule_every = kwargs['cfg']['schedule_every']
        self.schedule_inc   = kwargs['cfg']['schedule_increment']
        self.schedule_array = kwargs['cfg']['schedule_array']

        ### progressive quant levels 
        # self.ql_schedule_start = kwargs['cfg']['ql_schedule_start']
        # self.ql_schedule_every = kwargs['cfg']['ql_schedule_every']
        # self.ql_schedule_array = kwargs['cfg']['ql_schedule_array']

        self.samples        = kwargs['cfg']['samples']              # no. of samples per neuron (each side)
        
        ### quantization
        self.quant_func     = kwargs['quant_func'] if 'quant_func' in kwargs.keys() else None
        self.full_precision_and_range = kwargs['cfg']['full_precision_and_range']
        self.no_quant       = kwargs['cfg']['no_quant']
        self.hard_quant     = kwargs['cfg']['hard_quant'] if 'hard_quant' in kwargs['cfg'].keys() else False
        # self.binary         = kwargs['cfg']['binary']
        # self.with_zero      = kwargs['cfg']['with_zero']
        # self.first_step     = kwargs['cfg']['first_step']
        # self.quant_levels2  = kwargs['cfg']['quant_levels2']
        # self.phase_delta    = eval(kwargs['cfg']['phase_delta'])
        self.quant_levels   = kwargs['cfg']['quant_levels']
        self.lower_bound    = kwargs['cfg']['lower_bound'] if 'lower_bound' in kwargs['cfg'].keys() else 0
        self.upper_bounds   = kwargs['cfg']['upper_bounds'] if 'upper_bounds' in kwargs['cfg'].keys() else [1.99*np.pi]
        # self.quant_range    = kwargs['cfg']['quant_range']
        # self.start_y        = kwargs['cfg']['start_y']
        # self.end_y          = self.quant_levels - 1
        # self.quant_width    = (self.quant_range[1] - self.quant_range[0])/(self.quant_levels - 1) 
        # self.start_x        = self.quant_range[0] + self.quant_width/2

        #Default noise analysis modes
        self.fab_noise   = kwargs['cfg']['fab_noise']  #phase error
        self.dz_noise    = kwargs['cfg']['dz_noise']   #misalignment error (z axis)
        self.dx_noise    = kwargs['cfg']['dx_noise']   #neuron size error (lateral axis) 
        
        #Special error analysis params
        self.sf_noise       = kwargs['cfg']['sf_noise']
        self.default_sf     = kwargs['cfg']['default_sf']

        #Error analysis mode
        self.error_analysis = kwargs['cfg']['error_analysis']
        self.phase_error    = kwargs['cfg']['phase_error']
        
        self.RI_medium      = RI
        
        self.fake_quant     = kwargs['cfg']['fake_quant'] if 'fake_quant' in kwargs['cfg'].keys() else False
        self.inq            = kwargs['cfg']['inq']  ### incremental network quantization
        self.mlsq_inq       = kwargs['cfg']['mlsq_inq']  # combining MLSQ with INQ
        self.quant_interval = kwargs['cfg']['quant_interval']
        self.epochs         = kwargs['cfg']['epochs']
        self.cfg            = kwargs['cfg']
        self.weight_sf      = kwargs['cfg']['weight_sf']
        
        self.dsq_factor     = kwargs['cfg']['dsq_factor'] if 'dsq_factor' in kwargs['cfg'].keys() else 4

        self.G = get_G_with_window(self.n_i*self.samples, self.neuron_size/self.samples, self.delta_z, self.lambda_, w= self.w, RI = self.RI_medium).to(device)

        if weights!= None:
            amp_weights= weights['amp_weights']
            phase_weights= weights['phase_weights']    
            
        
        if (self.learn_type=='amp'):
            print('Learnable transmission coefficient: Amplitude only')
            if weights== None:
                self.amp_weights = nn.Parameter(torch.randn((self.n_i//self.weight_sf, self.n_i//self.weight_sf), dtype= torch.float), requires_grad= True)
                self.phase_weights= torch.zeros((self.n_i//self.weight_sf, self.n_i//self.weight_sf), dtype= torch.float).to(device)
            else:
                print('loading weights ... ')
                self.amp_weights = nn.Parameter(amp_weights, requires_grad= True)
                self.phase_weights= phase_weights.to(device)   
        elif (self.learn_type=='phase'):
            #print('Learnable transmission coefficient: Phase only')
            if weights== None:
                self.amp_weights = torch.ones((self.n_i//self.weight_sf, self.n_i//self.weight_sf), dtype= torch.float).to(device) *100000
                if self.cfg['no_quant']:
                    self.phase_weights= nn.Parameter(((self.cfg['upper_bounds'][-1] - self.cfg['lower_bound'])/(2*3)) * (torch.rand((self.n_i//self.weight_sf, self.n_i//self.weight_sf), dtype= torch.float) + 3) + self.cfg['lower_bound'], requires_grad= True)
                elif self.cfg['dsq'] or self.cfg['mlsq']:
                    self.phase_weights= nn.Parameter((self.cfg['upper_bounds'][-1] - self.cfg['lower_bound']) * torch.rand((self.n_i//self.weight_sf, self.n_i//self.weight_sf), dtype= torch.float) + self.cfg['lower_bound'], requires_grad= True)
                else:
                    self.phase_weights= nn.Parameter(torch.randn((self.n_i//self.weight_sf, self.n_i//self.weight_sf), dtype= torch.float), requires_grad= True)

            else:
                #print('loading weights ... ')
                self.amp_weights = torch.ones((self.n_i//self.weight_sf, self.n_i//self.weight_sf), dtype= torch.float).to(device) *100000 #amp_weights.to(device)
                self.phase_weights= nn.Parameter(phase_weights, requires_grad= True)
                
        elif (self.learn_type=='both'):
            print('Learnable transmission coefficient: Amplitude and Phase')
            if weights== None:
                self.phase_weights= nn.Parameter(torch.randn((self.n_i//self.weight_sf, self.n_i//self.weight_sf), dtype= torch.float), requires_grad= True)
                self.amp_weights = nn.Parameter(torch.randn((self.n_i//self.weight_sf, self.n_i//self.weight_sf), dtype= torch.float), requires_grad= True)
            else:
                print('loading weights ... ')
                self.phase_weights= nn.Parameter(phase_weights, requires_grad= True)
                self.amp_weights = nn.Parameter(amp_weights, requires_grad= True)
        else:
            #print('No learnable transmission coefficients')
            if weights== None:
                self.phase_weights= torch.zeros((self.n_i//self.weight_sf, self.n_i//self.weight_sf), dtype= torch.float).to(device)
                self.amp_weights = torch.ones((self.n_i//self.weight_sf, self.n_i//self.weight_sf), dtype= torch.float).to(device)*100000
            else:
                print('loading weights ... ')
                self.phase_weights= phase_weights.to(device)
                self.amp_weights = amp_weights.to(device)*100000  
        
        #### parameters for incremental network quantization
        self.T = torch.arange(torch.numel(self.phase_weights)).to(device)   # weight sampler
        self.mask = torch.ones(self.phase_weights.shape).to(device)         # mask for quantization
        self.inv_mask = ((self.mask == 0) * 1).to(device)                   # mask for training

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

    def mlsq_inq_scheduler(self, epoch):
        if epoch % self.quant_interval == 0:
            return 1
        else:
            return self.schedule_inc * (epoch % self.quant_interval)
    
    def inq_scheduler(self, epoch, batch_iter):
        ### scheduler for incremental network quantization
        if epoch == self.epochs - 1:
            if batch_iter == 0:
                return -1, True
            else:
                return -1, False
        elif batch_iter != 0:
            return 0, False
        elif epoch % self.quant_interval == 0:
            return epoch/self.quant_interval, True
        else:
            return 0, False
        
    def rigid_structure(self,structure_type, weights, device):
        n = weights.shape[0]

        if(structure_type == 'cross'):
            
            structure = torch.ones((n, n), dtype= torch.float).to(device)
            structure[n//2,:] = 0
            structure[:,n//2] = 0
    
            return structure*weights

        elif(structure_type == 'mesh'):
                
            structure = torch.ones((n, n), dtype= torch.float).to(device)
            structure[1::2,::n//4] = 0
            structure[::n//4,1::2] = 0
    
            return structure*weights

        elif(structure_type == 'cross_mesh'):
                
            structure = torch.ones((n, n), dtype= torch.float).to(device)
            structure[1::2,::n//4] = 0
            structure[::n//4,1::2] = 0
            structure[n//2,:] = 0
            structure[:,n//2] = 0
    
            return structure*weights
        elif (structure_type == 'sparse_pillers_with_cross'):
            structure = torch.ones((n, n), dtype= torch.float).to(device)

            structure[1::4,1::4] = 0
            structure[n//2-1,:] = 0
            structure[:,n//2-1] = 0
    
            return structure*weights

        elif (structure_type == 'sparse_pillers'):
            structure = torch.ones((n, n), dtype= torch.float).to(device)

            structure[1::4,1::4] = 0
    
            return structure*weights

        elif (structure_type =='hard_grid'):
            structure = torch.ones((n, n), dtype= torch.float).to(device)

            structure[1::1,::n//8] = 0
            structure[::n//8,1::1] = 0

            return structure*weights

        elif ('random' in structure_type):
            percentage = float(structure_type.split('_')[1])

            structure = (torch.rand(size=(n,n)) > percentage).float().to(device)

            return structure*weights
            
        else:
            raise ValueError("structure type not recognized")


    def __convert_raw_weights(self, epoch_n = None, batch_iter = None, device = None, cfg = None):
        if self.mlsq_inq:
            slope_factor = self.mlsq_inq_scheduler(epoch_n)
        elif self.cfg['dsq']:
            slope_factor = self.dsq_scheduler(epoch_n)
        else:
            slope_factor = self.scheduler(epoch_n)

        if(device == None):
            device = self.cfg['device']

        if self.learn_type == 'no':
            scaled_phase_weights = self.phase_weights
        elif self.full_precision_and_range:
            scaled_phase_weights = self.phase_weights
        elif self.no_quant:
            scaled_phase_weights = self.quant_func(self.phase_weights, 1, train = True)
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
                scaled_phase_weights = self.quant_func(self.phase_weights, k=slope_factor, train=True, noise = self.step_noise_value)
            else:
                scaled_phase_weights = self.quant_func(self.phase_weights, k=slope_factor, train=False, noise = self.step_noise_value)
             
        ##### Incremental Network Quantization (do quantization only once per epoch)
        elif (self.inq and self.training):
            i, requant   = self.inq_scheduler(epoch_n, batch_iter)
            
            if requant:
                # splitting the weights
                prob = torch.zeros(torch.numel(self.mask)).to(torch.float64)
                prob[self.T] = torch.tensor([1]).to(torch.float64)/len(self.T) #(2**i)/self.N
                
                if i == -1:
                    # print(f"epoch: {epoch_n}, batch: {batch_iter}, i: {i}, s1.1.last")
                    self.T = np.random.choice(torch.numel(self.mask), 0, replace=False, p=prob)
                else:
                    # print(f"epoch: {epoch_n}, batch: {batch_iter}, i: {i}, s1.1.nl")
                    # self.T = np.random.choice(torch.numel(self.phase_weights), int(torch.numel(self.phase_weights)//(2**(i+1))), replace=False, p=prob)
                    self.T = np.random.choice(torch.numel(self.mask), int(torch.numel(self.mask)*(1-0.1*(i+1))), replace=False, p=prob)

                # updating the mask
                self.mask = torch.ones(self.phase_weights.shape).to(device)
                torch.ravel(self.mask)[self.T] = 0
                self.inv_mask = (self.mask == 0) * 1
             
            if i == -1:
                scaled_phase_weights = self.quant_func(self.phase_weights, k=100, train=False, noise = self.step_noise_value)
            elif self.mlsq_inq:
                scaled_phase_weights = self.mask * self.quant_func(self.phase_weights, k=100, train=False, noise = self.step_noise_value) + self.inv_mask * self.quant_func(self.phase_weights, k=slope_factor, train=True, noise = self.step_noise_value)
            else:
                scaled_phase_weights = self.mask * self.quant_func(self.phase_weights, k=100, train=False, noise = self.step_noise_value) + self.inv_mask * self.phase_weights

        elif (self.inq and (not self.training)):
            scaled_phase_weights = self.quant_func(self.phase_weights, k=100, train=False, noise = self.step_noise_value)
        
        ##### Quantization using multi level sigmoid
        else:
            if ((self.training) and (epoch_n <= self.quant_after)):
                scaled_phase_weights = self.phase_weights
            elif ((self.training) and (epoch_n > self.quant_after)):
                scaled_phase_weights = self.quant_func(self.phase_weights, k=slope_factor, train=True, noise = self.step_noise_value)
            else:
                scaled_phase_weights = self.quant_func(self.phase_weights, k=100, train=False, noise = self.step_noise_value)
            
        return scaled_phase_weights, slope_factor


    def __introduce_noise_and_forced_structure(self, scaled_phase_weights, device = None, cfg = None):    
        
        # scaled_phase_weights = scaled_phase_weights + self.fab_noise * torch.randn(scaled_phase_weights.shape).to(device) #scale based on allowable unit phase shift
        if self.error_analysis:
            scaled_phase_weights = scaled_phase_weights + self.phase_error
        else:
            scaled_phase_weights = scaled_phase_weights + (2 * self.fab_noise * torch.rand(scaled_phase_weights.shape).to(device) - self.fab_noise) #add fabrication noise N_f ~ U[-self.fab_noise, self.fab_noise]
        
        del_z = cfg['eff_layer_dist']  #add interlayer distance noise N_f ~ U[-self.dz_noise, self.dz_noise]
        del_x = cfg['eff_neuron_size'] #add neuron size noise N_f ~ U[-self.dx_noise, self.dx_noise]
        
        self.G = get_G_with_window(self.n_i*self.samples, del_x/self.samples, del_z, self.lambda_, w= self.w, RI = self.RI_medium).to(device)
            
        if self.sf_noise:
            lower_bound = self.default_sf * 0.9
            upper_bound = self.default_sf * 1.1
            sf_ = np.arange(lower_bound, upper_bound+0.01, 0.01)
            scaled_phase_weights = scaled_phase_weights * self.default_sf / np.random.choice(sf_) 


        if self.cfg['rigid_structure'] != False:
            scaled_phase_weights = self.rigid_structure(self.cfg['rigid_structure'], scaled_phase_weights, device)

        return scaled_phase_weights

    def forward(self, input_e_field, epoch_n = None, batch_iter = None, cfg = None):
        device = input_e_field.device
        batch_size = input_e_field.shape[0]
        
        # update phase delta on batch basis
        # self.eff_phase_delta  = cfg['eff_phase_delta']  #resultant phase_step after adding noise N_f ~ U[-self.step_noise, self.step_noise]
        self.step_noise_value = cfg['step_noise_value']
        self.phase_error    = cfg['phase_error']
        self.error_analysis = cfg['error_analysis']

        scaled_phase_weights, slope_factor = self.__convert_raw_weights(epoch_n, batch_iter, device, cfg)

        scaled_phase_weights               = self.__introduce_noise_and_forced_structure(scaled_phase_weights, device, cfg)
        scaled_amp_weights = self.amp_weights

        if('weight_sf' in self.cfg.keys()):
            if(self.weight_sf > 1):
                pad_size = (self.cfg['img_size']*self.weight_sf//2 - self.cfg['img_size']//2)//2


                phase_padding = nn.ConstantPad2d(pad_size , 0)
                amp_padding   = nn.ConstantPad2d(pad_size , 1) 

                scaled_phase_weights = phase_padding(scaled_phase_weights)
                scaled_amp_weights   = amp_padding(self.amp_weights)


        # ts_1 = (torch.sigmoid(self.amp_weights) * torch.exp(1j*scaled_phase_weights)).view(1, self.n_i, self.n_i)
        ts_1 = (torch.sigmoid(scaled_amp_weights) * torch.exp(-1j*scaled_phase_weights)).view(1, self.n_i, self.n_i)
        ts = torch.repeat_interleave(torch.repeat_interleave(ts_1,self.samples,dim=2),self.samples,dim=1)
        
        input_e_field = input_e_field.view(batch_size, self.n_i*self.samples, self.n_i*self.samples)
        
        im_pad= torch.zeros((batch_size, self.w*self.n_i*self.samples, self.w*self.n_i*self.samples), dtype= torch.cfloat).to(device)
        im_pad[:, (self.w-1)*self.n_i*self.samples//2:self.n_i*self.samples+(self.w-1)*self.n_i*self.samples//2, (self.w-1)*self.n_i*self.samples//2:self.n_i*self.samples+(self.w-1)*self.n_i*self.samples//2]= input_e_field * ts

        A= torch.fft.fftshift(torch.fft.fft2(im_pad)) * self.G.view(1, self.w*self.n_i*self.samples, self.w*self.n_i*self.samples)
        B= torch.fft.ifft2(torch.fft.ifftshift(A))
        
        U = B[:, (self.w-1)*self.n_i*self.samples//2:(self.w-1)*self.n_i*self.samples//2+self.n_i*self.samples, (self.w-1)*self.n_i*self.samples//2:(self.w-1)*self.n_i*self.samples//2+self.n_i*self.samples]
        
        output_e_field = U

        return output_e_field, slope_factor, self.inv_mask

    
    
    
class d2nn_GS_layer(nn.Module):
    '''
        D2NN layer with Gumbel-Softmax quantization.
        
        Note: Do not pass the weights directly when inferencing. Initialize the model without any weights and use `model.load_state_dict()`
    '''
    
    def __init__(self, n_neurons_input, n_neurons_output, delta_z, lambda_, neuron_size, learn_type='both', device= 'cpu', window_size= 2, weights= None, RI = 1, **kwargs):
        super(d2nn_GS_layer, self).__init__()
        self.n_i               = n_neurons_input
        self.n_o               = n_neurons_output
        self.delta_z           = delta_z
        self.lambda_           = lambda_
        self.neuron_size       = neuron_size
        self.learn_type        = learn_type
        self.w                 = window_size

        self.schedule_start = kwargs['cfg']['schedule_start']
        self.samples        = kwargs['cfg']['samples']              # no. of samples per neuron (each side)
        
        ### quantization
        self.quant_levels   = kwargs['cfg']['quant_levels']
        self.lower_bound    = kwargs['cfg']['lower_bound'] if 'lower_bound' in kwargs['cfg'].keys() else 0
        self.upper_bounds   = kwargs['cfg']['upper_bounds'] if 'upper_bounds' in kwargs['cfg'].keys() else [1.99*np.pi]

        #Default noise analysis modes
        self.fab_noise   = kwargs['cfg']['fab_noise']  #phase error
        self.dz_noise    = kwargs['cfg']['dz_noise']   #misalignment error (z axis)
        self.dx_noise    = kwargs['cfg']['dx_noise']   #neuron size error (lateral axis) 
        
        #Special error analysis params
        self.sf_noise       = kwargs['cfg']['sf_noise']
        self.default_sf     = kwargs['cfg']['default_sf']

        #Error analysis mode
        self.error_analysis = kwargs['cfg']['error_analysis']
        self.phase_error    = kwargs['cfg']['phase_error']
        
        self.RI_medium      = RI
        
        self.epochs         = kwargs['cfg']['epochs']
        self.cfg            = kwargs['cfg']
        self.weight_sf      = kwargs['cfg']['weight_sf']

        self.G = get_G_with_window(self.n_i*self.samples, self.neuron_size/self.samples, self.delta_z, self.lambda_, w= self.w, RI = self.RI_medium).to(device)

        if weights!= None:
            amp_weights= weights['amp_weights']
            phase_weights= weights['phase_weights']
            
        self.quant_values = torch.linspace(self.lower_bound, self.upper_bounds[0], self.quant_levels[0]).to(device)
        
        if (self.learn_type=='phase'):
            #print('Learnable transmission coefficient: Phase only')
            if weights== None:
                self.amp_weights = torch.ones((self.n_i//self.weight_sf, self.n_i//self.weight_sf), dtype= torch.float).to(device) *100000
                self.phase_weights= nn.Parameter(nn.functional.gumbel_softmax(torch.randn((self.n_i//self.weight_sf, self.n_i//self.weight_sf, self.quant_levels[0]), dtype= torch.float), tau=10, hard=True), requires_grad= True)

            else:
                #print('loading weights ... ')
                self.amp_weights = torch.ones((self.n_i//self.weight_sf, self.n_i//self.weight_sf), dtype= torch.float).to(device) *100000 #amp_weights.to(device)
                
                phi = nn.functional.softmax(1 - torch.abs(phase_weights.unsqueeze(-1) - self.quant_values), dim=-1)
                self.phase_weights= nn.Parameter(phi, requires_grad= True)
        else:
            #print('No learnable transmission coefficients')
            if weights== None:
                self.phase_weights= torch.zeros((self.n_i//self.weight_sf, self.n_i//self.weight_sf), dtype= torch.float).to(device)
                self.amp_weights = torch.ones((self.n_i//self.weight_sf, self.n_i//self.weight_sf), dtype= torch.float).to(device)*100000
            else:
                print('loading weights ... ')
                self.phase_weights= phase_weights.to(device)
                self.amp_weights = amp_weights.to(device)*100000  

    def scheduler(self, epoch):
        return self.schedule_start - epoch * 0.5

    def __convert_raw_weights(self, epoch_n = None, batch_iter = None, device = None, cfg = None):
        
        slope_factor = self.scheduler(epoch_n)

        if self.learn_type == 'no':
            scaled_phase_weights = self.phase_weights
        else:
            scaled_phase_weights = torch.matmul(nn.functional.gumbel_softmax(self.phase_weights, tau=slope_factor, hard=True), self.quant_values)
            
        return scaled_phase_weights, slope_factor


#     def __introduce_noise_and_forced_structure(self, scaled_phase_weights, device = None, cfg = None):    
        
#         # scaled_phase_weights = scaled_phase_weights + self.fab_noise * torch.randn(scaled_phase_weights.shape).to(device) #scale based on allowable unit phase shift
#         if self.error_analysis:
#             scaled_phase_weights = scaled_phase_weights + self.phase_error
#         else:
#             scaled_phase_weights = scaled_phase_weights + (2 * self.fab_noise * torch.rand(scaled_phase_weights.shape).to(device) - self.fab_noise) #add fabrication noise N_f ~ U[-self.fab_noise, self.fab_noise]
        
#         del_z = cfg['eff_layer_dist']  #add interlayer distance noise N_f ~ U[-self.dz_noise, self.dz_noise]
#         del_x = cfg['eff_neuron_size'] #add neuron size noise N_f ~ U[-self.dx_noise, self.dx_noise]
        
#         self.G = get_G_with_window(self.n_i*self.samples, del_x/self.samples, del_z, self.lambda_, w= self.w, RI = self.RI_medium).to(device)
            
#         if self.sf_noise:
#             lower_bound = self.default_sf * 0.9
#             upper_bound = self.default_sf * 1.1
#             sf_ = np.arange(lower_bound, upper_bound+0.01, 0.01)
#             scaled_phase_weights = scaled_phase_weights * self.default_sf / np.random.choice(sf_) 

#         return scaled_phase_weights

    def forward(self, input_e_field, epoch_n = None, batch_iter = None, cfg = None):
        device = input_e_field.device
        batch_size = input_e_field.shape[0]
        
        # update phase delta on batch basis
        # self.eff_phase_delta  = cfg['eff_phase_delta']  #resultant phase_step after adding noise N_f ~ U[-self.step_noise, self.step_noise]
        self.step_noise_value = cfg['step_noise_value']
        
        scaled_phase_weights, slope_factor = self.__convert_raw_weights(epoch_n, batch_iter, device, cfg)

        # scaled_phase_weights               = self.__introduce_noise_and_forced_structure(scaled_phase_weights, device, cfg)
        scaled_amp_weights = self.amp_weights

        if('weight_sf' in self.cfg.keys()):
            if(self.weight_sf > 1):
                pad_size = (self.cfg['img_size']*self.weight_sf//2 - self.cfg['img_size']//2)//2


                phase_padding = nn.ConstantPad2d(pad_size , 0)
                amp_padding   = nn.ConstantPad2d(pad_size , 1) 

                scaled_phase_weights = phase_padding(scaled_phase_weights)
                scaled_amp_weights   = amp_padding(self.amp_weights)


        # ts_1 = (torch.sigmoid(self.amp_weights) * torch.exp(1j*scaled_phase_weights)).view(1, self.n_i, self.n_i)
        ts_1 = (torch.sigmoid(scaled_amp_weights) * torch.exp(-1j*scaled_phase_weights)).view(1, self.n_i, self.n_i)
        ts = torch.repeat_interleave(torch.repeat_interleave(ts_1,self.samples,dim=2),self.samples,dim=1)
        
        input_e_field = input_e_field.view(batch_size, self.n_i*self.samples, self.n_i*self.samples)
        
        im_pad= torch.zeros((batch_size, self.w*self.n_i*self.samples, self.w*self.n_i*self.samples), dtype= torch.cfloat).to(device)
        im_pad[:, (self.w-1)*self.n_i*self.samples//2:self.n_i*self.samples+(self.w-1)*self.n_i*self.samples//2, (self.w-1)*self.n_i*self.samples//2:self.n_i*self.samples+(self.w-1)*self.n_i*self.samples//2]= input_e_field * ts

        A= torch.fft.fftshift(torch.fft.fft2(im_pad)) * self.G.view(1, self.w*self.n_i*self.samples, self.w*self.n_i*self.samples)
        B= torch.fft.ifft2(torch.fft.ifftshift(A))
        
        U = B[:, (self.w-1)*self.n_i*self.samples//2:(self.w-1)*self.n_i*self.samples//2+self.n_i*self.samples, (self.w-1)*self.n_i*self.samples//2:(self.w-1)*self.n_i*self.samples//2+self.n_i*self.samples]
        
        output_e_field = U

        return output_e_field, slope_factor, 0