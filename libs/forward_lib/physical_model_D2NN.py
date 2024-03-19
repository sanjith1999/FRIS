import torch
import numpy as np
import scipy.special
from torch import nn
import libs.forward_lib.visualizer as vs
import copy
import libs.forward_lib.angular_spectrum as AS
from libs.forward_lib.modules.d2nn_models_new import *


# noinspection PyAttributeOutsideInit
class PhysicalModel:
    """
     Class: represents the whole forward process of a single photon microscopy

     self.init_psf(), self.D2NN.initialize_fields() should be called to initialize patterns and PSF
    """

    # Class Variables
    lambda_ = 532.0 / 1000  # um
    NA = .8
    r_index = 1
    # dx, dy, dz = 0.25, 0.25, 0.25  # um
    dx, dy, dz = 1.0, 1.0, 1.0  # um
    w = 2

    def __init__(self, nx, ny, nz, n_patterns, dd_factor=1, n_planes=1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.n_patterns = n_patterns
        self.dd_factor = dd_factor
        self.n_planes = n_planes
        self.m_planes = [(i * nz) // (n_planes + 1) for i in range(1, n_planes + 1)]
        self.device = device
        self.D2NN = D2NN_patterns(self.nx, self.ny, self.dx, self.dy, self.n_patterns, self.device)

    def __str__(self):
        desc = ""
        desc += "Forward Model Specifications\n----------------------------------------------\n"
        desc += f"Space Dimension \t\t: {self.nx * self.dx}um × {self.ny * self.dy}um × {self.nz * self.dz}um\n"
        desc += f"voxel Size \t\t\t: {self.dx}um × {self.dy}um × {self.dz}um\n"
        desc += f"D2NN Patterns \t\t\t: {self.n_patterns}\n"
        desc += f"Measurement Plane\t\t: {self.m_planes}\n"
        desc += f"Detector Pool size \t\t: {self.dd_factor}×{self.dd_factor}\n"
        desc += f"Computational Device \t\t: {self.device}"
        return desc

    def init_psf(self):
        """
        Method: calculate the point spread function and interpret both excitation and emission parts
        """
        psf = psf_model(self.NA, self.r_index, self.lambda_, self.dx, self.dy, self.dz, self.nx, self.ny, self.nz).to(self.device)
        self.exPSF_3D = psf().detach().permute(0, 3, 1, 2)
        self.emPSF_3D = self.exPSF_3D.abs().square().sum(dim=0).sqrt().unsqueeze(dim=0)

    def load_psf(self, IT):
        """
        Method: store and recover PSF
        """
        psf = (torch.load(f"./data/matrices/field/PSF_{IT}.pt")['matrix']).to(self.device)  # Manual extra-care should be taken to match parameters
        self.exPSF_3D = psf().detach().permute(0, 3, 1, 2)
        self.emPSF_3D = self.exPSF_3D.abs().square().sum(dim=0).sqrt().unsqueeze(dim=0)

    def propagate_D2NN(self, p_no=1):
        """
        Method: forwarding D2NN field to object space
        """
        aNs = AS.d2nnASwWindow_layer(self.nx, self.ny, self.dx, self.dy, self.dz, self.lambda_, window_size=8)
        H1 = torch.zeros(self.nz, self.nx, self.ny, dtype=torch.complex64)

        # shift = self.D2NN.cfg['delta_z']/2 + self.D2NN.cfg['out_dist']
        shift = 0
        for i in range(0, self.nz):
            prop_dist = (i-self.nz//2)*self.dz - shift
            aNs.find_transfer_function(prop_dist, mask_factor_=self.NA**2)
            output_field = aNs.forward(self.D2NN.ht_2D_list[p_no-1].cpu().unsqueeze(0))
            H1[i] = output_field.detach()
        self.H2 = torch.abs(H1)

    def propagate_impulse(self, cor=(0, 0, 0)):
        """
        Method: Propagating an impulse at certain location of the object to the detector
        """
        ix, iy, iz = cor
        cx, cy = self.nx // 2, self.ny // 2
        det_Y = torch.zeros(self.nx, self.ny).to(self.device).float()

        i_iz = self.nz - 1 - iz
        l_ix, l_iy = min(cx, ix), min(cy, iy)
        r_ix, r_iy = min(self.nx - ix, cx), min(self.ny - iy, cy)

        det_Y[ix - l_ix:ix + r_ix, iy - l_iy:iy + r_iy] = self.H2[iz, ix, iy] * self.emPSF_3D[0, i_iz, cx - l_ix: cx + r_ix, cy - l_iy:cy + r_iy]
        scale_factor = (1 / self.dd_factor, 1 / self.dd_factor)
        det_Y = torch.nn.functional.interpolate(det_Y.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor, mode='area').squeeze()
        return det_Y

    def propagate_patch(self, X):
        """
        Method: Propagating a patch to detector
        """
        H3 = X * self.H2
        Y = conv_3D(self.emPSF_3D, H3, self.w).abs()[0]  # field around the detector
        det_Y = Y[self.m_planes, :, :]
        scale_factor = (1, 1 / self.dd_factor, 1 / self.dd_factor) if len(det_Y.shape) == 3 else (1 / self.dd_factor, 1 / self.dd_factor)
        det_Y = nn.functional.interpolate(det_Y.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor, mode='area').squeeze()
        return det_Y

    def propagate_object(self, X, p_no=1, verbose=0):
        """
        Method: forward process on an object
        """
        aNs = AS.d2nnASwWindow_layer(self.nx, self.ny, self.dx, self.dy, self.dz, self.lambda_, window_size=8)
        H1 = torch.zeros(self.nz, self.nx, self.ny, dtype=torch.complex64)
        
        # shift = self.D2NN.cfg['delta_z']/2 + self.D2NN.cfg['out_dist']
        shift = 0
        for i in range(0, self.nz):
            prop_dist = (i-self.nz//2)*self.dz - shift
            aNs.find_transfer_function(prop_dist, mask_factor_=self.NA**2)
            output_field = aNs.forward(self.D2NN.ht_2D_list[p_no-1].cpu().unsqueeze(0))
            H1[i] = output_field.detach()
        H2 = torch.abs(H1)

        H3 = X * H2
        Y = conv_3D(self.emPSF_3D, H3, self.w).abs()[0]  # field around the detector
        det_Y = Y[self.m_planes, :, :]
        scale_factor = (1, 1 / self.dd_factor, 1 / self.dd_factor) if len(det_Y.shape) == 3 else (1 / self.dd_factor, 1 / self.dd_factor)
        det_Y = nn.functional.interpolate(det_Y.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor, mode='area').squeeze()

        # Visualizing Stuff
        if verbose > 0:
            det_R = det_Y.detach().cpu().numpy()
            if verbose > 1:
                if verbose > 2:
                    coherent_out = H2.detach().cpu().numpy()
                    vs.show_planes(coherent_out, title=f'H2', N_z=self.nz)
                X_D = X[0].detach().cpu().numpy()
                vs.show_planes(X_D, title=f"Object", N_z=self.nz)
                Y_D = Y.detach().cpu().numpy()
                vs.show_planes(Y_D, title=f'Y', N_z=self.nz)
            if self.n_planes == 1:
                vs.show_image(det_R, f"M-Plane: {self.m_planes[0]}", (3, 3))
            else:
                vs.show_images(images=[det_r for det_r in det_R], titles=[f"M-Plane: {i}" for i in self.m_planes], figsize=(3 * self.n_planes, 3), cols=self.n_planes)
        if verbose > 3:
            print(f"Shapes of vectors\n exPSF: {self.exPSF_3D.shape}\n H1: {H1.shape}\n H2: {H2.shape}\n emPSF: {self.emPSF_3D.shape}\n H3: {H3.shape}")

        return det_Y

    def extended_propagation(self, specimen, verbose=False):
        """
        Method: Perform the forward process operation for multiple D2NN patterns
        """
        Y = torch.tensor([]).to(self.device)
        for p in range(1, self.n_patterns + 1):
            Yi = self.propagate_object(specimen, p_no=p, verbose=verbose)
            Yi_flatten = Yi.flatten()
            Y = torch.cat((Y, Yi_flatten), dim=0)
        return Y



# noinspection PyAttributeOutsideInit
class D2NN_patterns:
    """
    Class: Store multiple D2NN fields
    """

    def __init__(self, nx, ny, dx, dy, n_patterns, device):
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy
        self.n_patterns = n_patterns
        self.device = device
        self.D2NN_model = None

        self.cfg = {
            'device': self.device,
            'model': 'd2nnASwWindow',
            'RI_Gel': 1,
            'lambda_': PhysicalModel.lambda_ * 1e-6,
            'n_layers': 2,
            'delta_z': 3.3e-06,
            'in_dist': 3.3e-06,
            'out_dist': 3.3e-06, 
            'neuron_size': PhysicalModel.dx * 1e-6,
            
            'img_size': self.nx,
            'shrink_factor': 1,

            'torch_seed': 5, # changing this will change the random weights of the D2NN
            'learn_type': 'phase',

            'quant_after': -1,
            'schedule_start': 1,
            'schedule_increment': 0.05,
            'schedule_every': 1,
            
            'full_precision_and_range': True,
            'unwrapped_phase': True,
            'pretrained': False,
            'no_quant': False,
            'mlsq': False,
            'dsq': False,
            'gs_quant': False,
            'hard_quant': False,
            'inq': False,
            'mlsq_inq': False,
            'fake_quant': False,
            
            'quant_levels': [2, 7],
            'lower_bound': 0,
            'upper_bounds': [0.5340707511102649, 3.267256359733385],
            'learn_u': [False, False],
            'alpha': 0,
            'learn_schedule': False,
            'mlsq_reg': [0.01, 0.1],
            'reg_schedule': ['pow2', 5],
            
            'dsq_alpha': 0.2,
            'dsq_factor': 10,
            'dsq_regularize': [0.01, 0.5],
            'dsq_temp_const': False,
            'dsq_temp_learn': True,
            'int_ramp': 'no',
            
            'window_size': 4,
            'biasOnoise': 0,
            'energy_type': 'passive',
            'samples': 1,
            'local_norm_amp': False,

            'photon_count': 1,

            'fab_noise': 0,
            'dz_noise': 0,
            'dx_noise': 0,
            'step_noise': 0,
            'rigid_structure': False,
            'sf_noise': False,
            'default_sf': 'nan',
            'error_analysis': False,
            'phase_error': 0,
            'dz_error': 0,
            'dx_error': 0,
            
            'quant_interval': None,
            'epochs': 1,
            'weight_sf': 1,
            'output_scalebias_matrix': False,
            'output_scalebias_for_INTENSITY': False,
            'output_scale': 1.0,
            'output_bias': 0.0,
            'output_scale_learnable': False,
            'output_bias_learnable': False,
        }

        scheduler = torch.arange(self.cfg['schedule_start'], 1+self.cfg['schedule_start']+(10-self.cfg['quant_after'])//self.cfg['schedule_every']*\
                                self.cfg['schedule_increment'], self.cfg['schedule_increment'])
        self.cfg['schedule_array'] = scheduler

    def d2nn_propagation(self, model, input_field):
        device = self.cfg['device']
        img_size   = self.cfg['img_size']
        samples    = self.cfg['samples']
        shrinkFactor = self.cfg['shrink_factor'] if 'shrink_factor' in self.cfg.keys() else 1
        if(shrinkFactor!=1):
            csize = int((img_size*samples)/shrinkFactor)
            spos  = int((img_size*samples - csize)/2)
            epos  = spos + csize
        else:
            spos = 0
            epos = img_size*samples

        if self.cfg['error_analysis']:
            self.cfg['eff_neuron_size'] = self.cfg['neuron_size'] + self.cfg['dx_error'] 
            self.cfg['eff_layer_dist']  = self.cfg['delta_z'] + self.cfg['dz_error']
        else:
            self.cfg['eff_neuron_size']  =  self.cfg['neuron_size'] + (2 * self.cfg['dx_noise'] * torch.rand(1).item() - self.cfg['dx_noise']) #add neuron size noise N_f ~ U[-self.dx_noise, self.dx_noise]
            self.cfg['eff_layer_dist']   =  self.cfg['delta_z']     + (2 * self.cfg['dz_noise'] * torch.rand(1).item() - self.cfg['dz_noise']) #add interlayer distance noise N_f ~ U[-self.dz_noise, self.dz_noise]
            self.cfg['step_noise_value'] =  (2 * self.cfg['step_noise'] * torch.rand(1).item() - self.cfg['step_noise']) #phase step noise N_f ~ U[-self.dp_noise, self.dp_noise]

        ground_truth = (input_field.unsqueeze(0)).to(device)
        model.eval()
        with torch.no_grad():
            pred_img, out_bias,out_scale, temp, mask_list = model(ground_truth, 0, 0, self.cfg)
            pred_img = pred_img[:,spos:epos,spos:epos]
        return pred_img
    
    def initialize_D2NN_model(self):
        """
        Method: Initialize a D2NN model with layers
        """ 
        torch.manual_seed(self.cfg['torch_seed']) 
        model = eval(self.cfg['model'])(self.cfg).to(self.cfg['device'])
        self.D2NN_model = model

    def get_D2NN_output_field(self, input_field):
        """
        Method: Initialize a D2NN field on focal plane
        """ 
        field = torch.tensor([]).to(self.cfg['device'])
        field = self.d2nn_propagation(self.D2NN_model, input_field) 
        return field.squeeze()

    def initialize_D2NN_fields(self, IT=-1):
        """
        Method: form a list of patterns that contain m random initializations
        """
        self.initialize_D2NN_model()
        self.ht_2D_list = []
        data_to_save = {}
        path_to_save = f"./data/matrices/D2NN/base_{IT}.pt"
        delta = torch.pi/100
        thetas_x = torch.linspace(0+delta, torch.pi-delta, self.n_patterns//2)
        thetas_y = torch.linspace(0+delta, torch.pi-delta, self.n_patterns - self.n_patterns//2)
        k = 2 * torch.pi / PhysicalModel.lambda_
        # change x angle direction
        for i in range(self.n_patterns//2):
            phi_x = torch.linspace(0, self.nx*k*self.dx*torch.cos(thetas_x[i]), self.nx)
            phi_xy = torch.tile(phi_x, (self.nx,1))
            input_field_i = torch.ones(self.nx, self.nx) * torch.exp(1j * phi_xy)
            ht_2D = self.get_D2NN_output_field(input_field_i)
            self.ht_2D_list.append(ht_2D)
        # change y angle direction
        for i in range(self.n_patterns - self.n_patterns//2):
            phi_y = torch.linspace(0, self.nx*k*self.dy*torch.cos(thetas_y[i]), self.nx)
            phi_y = phi_y.unsqueeze(1)
            phi_xy = torch.tile(phi_y, (1, self.nx))
            input_field_i = torch.ones(self.nx, self.nx) * torch.exp(1j * phi_xy)
            ht_2D = self.get_D2NN_output_field(input_field_i)
            self.ht_2D_list.append(ht_2D)
        if IT != -1:
            torch.save(data_to_save, path_to_save)

    def visualize_patterns(self):
        """
        Method: visualizing the D2NN patterns
        """
        if self.n_patterns == 1:
            vs.show_image(torch.abs(self.ht_2D_list[0]).cpu().detach(), "Pattern", fig_size=(3, 3))
        elif self.n_patterns > 1:
            vs.show_images([torch.abs(ht_2D).cpu().detach() for ht_2D in self.ht_2D_list], titles=[f"Pattern {i + 1}" for i in range(self.n_patterns)], cols=self.n_patterns, figsize=(6, 3))
        else:
            print("Nothing to visualize...!")


# Helper Function
def calculate_phi(NPXLS):
    N = (NPXLS - 1) / 2
    A = torch.arange(-N, N + 1)

    XX = torch.unsqueeze(A, dim=0)
    YY = -torch.unsqueeze(A, dim=1)

    X = torch.tile(XX, (NPXLS, 1))
    Y = torch.tile(YY, (1, NPXLS))

    phi = torch.atan2(Y, X)
    return phi

# 3D Convolution
def conv_3D(PSF_3D, H, w=1):
    Bp, nz, nx, ny = PSF_3D.shape
    Bh, _, _, _ = H.shape
    PSF_3D_PAD = torch.zeros(Bp, w * nz, w * nx, w * ny).type(torch.complex64).to(PSF_3D.device)
    H_PAD = torch.zeros(Bh, w * nz, w * nx, w * ny).type(torch.complex64).to(H.device)

    PSF_3D_PAD[:, (w - 1) * nz // 2:nz + (w - 1) * nz // 2, (w - 1) * nx // 2:nx + (w - 1) * nx // 2, (w - 1) * ny // 2:ny + (w - 1) * ny // 2] = PSF_3D
    H_PAD[:, (w - 1) * nz // 2:nz + (w - 1) * nz // 2, (w - 1) * nx // 2:nx + (w - 1) * nx // 2, (w - 1) * ny // 2:ny + (w - 1) * ny // 2] = H

    Ht_fft = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(H_PAD, dim=(-3, -2, -1)), dim=(-3, -2, -1)), dim=(-3, -2, -1))
    PSF_fft = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(PSF_3D_PAD, dim=(-3, -2, -1)), dim=(-3, -2, -1)), dim=(-3, -2, -1))
    conv_PSF_H_PAD = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(PSF_fft * Ht_fft, dim=(-3, -2, -1)), dim=(-3, -2, -1)), dim=(-3, -2, -1))

    conv_PSF_H = conv_PSF_H_PAD[:, (w - 1) * nz // 2:(w - 1) * nz // 2 + nz, (w - 1) * nx // 2:(w - 1) * nx // 2 + nx, (w - 1) * ny // 2:(w - 1) * ny // 2 + ny]
    return conv_PSF_H


# PSF Model as a neural node
# noinspection PyAttributeOutsideInit
class psf_model(nn.Module):
    def __init__(self, NA, Rindex, lambda_, dx, dy, dz, Nx, Ny, Nz):
        super().__init__()
        self.NA = NA
        self.Rindex = Rindex
        self.lambda_ = lambda_
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Ntheta = 400  # n grid in polar angle

        self.abberations = torch.nn.Parameter(
            torch.zeros((self.Ntheta,)).float())

        self.init_psf_params()

    def init_psf_params(self):
        self.alpha = np.arcsin(self.NA / self.Rindex)

        self.x = self.dx * torch.arange(-self.Nx // 2 + 1, self.Nx // 2 + 1)
        self.y = self.dy * torch.arange(-self.Ny // 2 + 1, self.Ny // 2 + 1)
        self.z = self.dz * torch.arange(-self.Nz // 2 + 1, self.Nz // 2 + 1)

        self.dtheta = self.alpha / self.Ntheta
        self.theta = torch.arange(0, self.Ntheta) * self.dtheta

        assert self.Nx == self.Ny, "self.Nx != self.Ny -> self.Phi calculation wrong !!!"
        self.Phi = calculate_phi(self.Nx)
        self.A = np.pi / self.lambda_

        self.X, self.Y, self.THETA = torch.meshgrid(self.x, self.y, self.theta, indexing='ij')

        V = (2 * np.pi / self.lambda_) * \
            torch.sqrt(self.X ** 2 + self.Y ** 2).numpy()  # k.r
        J0 = torch.from_numpy(scipy.special.jv(
            0, V * np.sin(self.THETA.numpy())))
        J1 = torch.from_numpy(scipy.special.jv(
            1, V * np.sin(self.THETA.numpy())))
        J2 = torch.from_numpy(scipy.special.jv(
            2, V * np.sin(self.THETA.numpy())))

        self.Func0 = torch.sqrt(torch.cos(
            self.THETA)) * torch.sin(self.THETA) * (1 + torch.cos(self.THETA)) * J0
        self.Func1 = torch.sqrt(torch.cos(self.THETA)) * \
                     (torch.sin(self.THETA) ** 2) * J1
        self.Func2 = torch.sqrt(torch.cos(
            self.THETA)) * torch.sin(self.THETA) * (1 - torch.cos(self.THETA)) * J2

        self.U = 2 * np.pi / self.lambda_ * self.z  # k.z

    def forward(self):
        device = self.abberations.device
        ABBR = torch.tile(self.abberations,
                          (self.x.shape[0], self.y.shape[0], 1))

        Func0 = self.Func0.to(device) * torch.exp(1j * ABBR)
        Func1 = self.Func1.to(device) * torch.exp(1j * ABBR)
        Func2 = self.Func2.to(device) * torch.exp(1j * ABBR)

        PSF_3D = torch.zeros((3, self.Nx, self.Ny, self.Nz),
                             dtype=torch.cfloat).to(device)

        for k in range(len(self.U)):
            Func3_atThisU = torch.exp(-1j *
                                      self.U[k] * torch.cos(self.THETA)).to(device)

            I0 = torch.trapz(y=Func0 * Func3_atThisU, x=self.theta.to(device), axis=2)  # axis: alpha
            I1 = torch.trapz(y=Func1 * Func3_atThisU, x=self.theta.to(device), axis=2)  # axis: alpha
            I2 = torch.trapz(y=Func2 * Func3_atThisU, x=self.theta.to(device), axis=2)  # axis: alpha

            Ex = 1j * self.A * (I0 + I2 * torch.cos(2 * self.Phi.to(device)))
            Ey = 1j * self.A * I2 * torch.sin(2 * self.Phi.to(device))
            Ez = (-2) * self.A * I1 * torch.cos(self.Phi.to(device))

            # Es = 1j * A * I0 ## scalar approx
            PSF_3D[0, :, :, k] = Ex
            PSF_3D[1, :, :, k] = Ey
            PSF_3D[2, :, :, k] = Ez

        return PSF_3D
