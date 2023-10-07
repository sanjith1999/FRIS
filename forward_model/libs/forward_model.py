import torch
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from torch import nn


# Initializing important parameters
def init_parameters(NA_, Rindex_, lambda_, dx_, Nx_, Ny_, Nz_, patch_size_=None, verbose=False, device_=None):
    global Nx, Ny, Nz, exPSF_3D, emPSF_3D, device

    Nx, Ny, Nz = Nx_, Ny_, Nz_
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        psf = psf_model(NA_, Rindex_, lambda_, dx_, Nx, Ny, Nz).to(device)
        exPSF_3D = psf().detach().permute(0, 3, 1, 2)
        emPSF_3D = (exPSF_3D.abs()**2).sum(dim=0).unsqueeze(dim=0)  # IPSF
        print("Sucessfully Initialized Point Spread Function...!!!")
    except:
        print("Point Spread Function Initialization Failed...!!!")


# Initializing DMD Pattern : Simple Model
def init_DMD(patch_size_=None, verbose=False):
    global ht_2D

    if (patch_size_):
        patch_size = patch_size_
    else:
        patch_size = max(min(Nx, Ny)//20, 1)

    try:
        ht_2D = (torch.randn(Nx//patch_size + 1, Ny//patch_size + 1) > 0).float()
        ht_2D = ht_2D.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)[:Nx, :Ny]
        if verbose:
            show_image(ht_2D, "Excitation Pattern", fig_size=(5, 5))

    except:
        print("DMD Pattern Initialization Failed...!!!")


# Initializing DMD Patterns : Extended Model
def init_DMD_patterns(m):
    global Ht_2D_list
    Ht_2D_list = []
    for _ in range(m):
        init_DMD()
        Ht_2D_list.append(ht_2D)


# Forward model of a single measurement
def forward_model(X, Ht_2D=None, verbose=0):
    ht_3D = torch.zeros(1, Nz, Nx, Ny).float().to(device)
    if Ht_2D is None:
        ht_3D[:, Nz // 2] = ht_2D
    else:
        ht_3D[:, Nz//2] = Ht_2D
    H1 = conv_3D(exPSF_3D, ht_3D)
    H2 = (H1.abs()**2).sum(dim=0)

    H3 = X * H2
    Y = conv_3D(emPSF_3D, H3).abs()[0]
    det_Y = Y[round(Nz/2), :, :]

    ####### DETACH IMPORTANT OBJECTS TO VISUALIZE ##########
    # Excitation Pattern Convolved with Coherent PSF : H2
    coherent_out = H2.detach().cpu().numpy()
    # Normalized Image : X
    I = X[0].detach().cpu().numpy()

    # Image at Camera: Y
    R = Y.detach().cpu().numpy()

    # Image at Detector: ~Y
    det_R = det_Y.detach().cpu().numpy()

    """     
    verbose = 0 -> No Visualization
    verbose = 1 -> Visualize Detected Image
    verbose = 2 -> Object & Y
    verbose = 3 -> Excitation Pattern & H2
    """
    if verbose > 0:
        if verbose > 1:
            if verbose > 2:
                show_image(ht_2D, "Excitation Pattern", fig_size=(3, 3))
                show_planes(coherent_out, title=f'H2', N_z=Nz)
            show_planes(I, title=f"Object", N_z=Nz)
            show_planes(R, title=f'Y', N_z=Nz)
        show_image(det_R, "Detected Image", (3, 3))

    return det_Y


# Extended forward model
def extended_forward_model(X,verbose=False):
    Y = torch.tensor([]).to(device)
    for Ht_2D in Ht_2D_list:
        Yi = forward_model(X, Ht_2D,verbose=verbose)
        Yi_flatten = Yi.flatten()
        Y = torch.cat((Y, Yi_flatten), dim=0)
    return Y


# Initializng parameters for One Shot Model
def init_one_shot(m):
    global A
    try:
        A = torch.zeros(Nx*Ny*m,Nx*Ny*Nz).float().to(device)
        I = torch.zeros(1, Nz, Nx, Ny).float().to(device)
        for i_z in range(Nz):
            for i_x in range(Nx):
                for i_y  in range(Ny):
                    I[0,i_z,i_x,i_y] = 1
                    A [:,i_z*Ny*Nx+ i_x*Ny+i_y] = extended_forward_model(I)
                    I[0,i_z,i_x,i_y] = 0
        print("Matrix A is intialized sucessfully...!!!")
    except:
        print("Failed to Initialize A...!!!")


# Helper Function
def calculate_phi(NPXLS):
    N = (NPXLS-1)/2
    A = torch.arange(-N, N+1)

    XX = torch.unsqueeze(A, dim=0)
    YY = -torch.unsqueeze(A, dim=1)

    X = torch.tile(XX, (NPXLS, 1))
    Y = torch.tile(YY, (1, NPXLS))

    phi = torch.atan2(Y, X)
    return phi


# 3D Convolution
def conv_3D(PSF_3D, H):

    Ht_fft = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(
        H, dim=(-3, -2, -1)), dim=(-3, -2, -1)), dim=(-3, -2, -1))
    PSF_fft = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(
        PSF_3D, dim=(-3, -2, -1)), dim=(-3, -2, -1)), dim=(-3, -2, -1))
    conv_PSF_H = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(
        PSF_fft * Ht_fft, dim=(-3, -2, -1)), dim=(-3, -2, -1)), dim=(-3, -2, -1))

    return conv_PSF_H


# PSF Model as a neural node
class psf_model(nn.Module):
    def __init__(self, NA, Rindex, lambda_, dx, Nx, Ny, Nz):
        super().__init__()
        self.NA = NA
        self.Rindex = Rindex
        self.lambda_ = lambda_
        self.dx = dx
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Ntheta = 400  # n grid in polar angle

        self.abberations = torch.nn.Parameter(
            torch.zeros((self.Ntheta, )).float())

        self.init_psf_params()

    def init_psf_params(self):
        self.alpha = np.arcsin(self.NA / self.Rindex)
        self.dy = self.dx
        self.dz = self.dx

        self.x = self.dx * torch.arange(-self.Nx//2+1, self.Nx//2+1)
        self.y = self.dy * torch.arange(-self.Ny//2+1, self.Ny//2+1)
        self.z = self.dz * torch.arange(-self.Nz//2+1, self.Nz//2+1)

        self.dtheta = self.alpha / self.Ntheta
        self.theta = torch.arange(0, self.Ntheta)*self.dtheta

        assert self.Nx == self.Ny, "self.Nx != self.Ny -> self.Phi calculation wrong !!!"
        self.Phi = calculate_phi(self.Nx)
        self.A = np.pi / self.lambda_

        self.X, self.Y, self.THETA = torch.meshgrid(self.x, self.y, self.theta)

        V = (2*np.pi / self.lambda_) * \
            torch.sqrt(self.X**2 + self.Y**2).numpy()  # k.r
        J0 = torch.from_numpy(scipy.special.jv(
            0, V * np.sin(self.THETA.numpy())))
        J1 = torch.from_numpy(scipy.special.jv(
            1, V * np.sin(self.THETA.numpy())))
        J2 = torch.from_numpy(scipy.special.jv(
            2, V * np.sin(self.THETA.numpy())))

        self.Func0 = torch.sqrt(torch.cos(
            self.THETA)) * torch.sin(self.THETA) * (1 + torch.cos(self.THETA)) * J0
        self.Func1 = torch.sqrt(torch.cos(self.THETA)) * \
            (torch.sin(self.THETA)**2) * J1
        self.Func2 = torch.sqrt(torch.cos(
            self.THETA)) * torch.sin(self.THETA) * (1 - torch.cos(self.THETA)) * J2

        self.U = 2*np.pi / self.lambda_ * self.z  # k.z

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

            I0 = torch.trapz(y=Func0 * Func3_atThisU,
                             x=self.theta.to(device), axis=2)  # axis: alpha
            I1 = torch.trapz(y=Func1 * Func3_atThisU,
                             x=self.theta.to(device), axis=2)  # axis: alpha
            I2 = torch.trapz(y=Func2 * Func3_atThisU,
                             x=self.theta.to(device), axis=2)  # axis: alpha

            Ex = 1j * self.A * (I0 + I2 * torch.cos(2*self.Phi.to(device)))
            Ey = 1j * self.A * I2 * torch.sin(2*self.Phi.to(device))
            Ez = (-2) * self.A * I1 * torch.cos(self.Phi.to(device))

            # Es = 1j * A * I0 ## scalar approx
            PSF_3D[0, :, :, k] = Ex
            PSF_3D[1, :, :, k] = Ey
            PSF_3D[2, :, :, k] = Ez

        return PSF_3D


# Visualizing PSF
def show_psf(PSF_3D):
    _, Nx, Ny, Nz = PSF_3D.shape

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(PSF_3D[0, :, :, Nz//2].abs())
    plt.title('x-axis')
    plt.colorbar(fraction=0.025)

    plt.subplot(1, 3, 2)
    plt.imshow(PSF_3D[1, :, :, Nz//2].abs())
    plt.title('y-axis')
    plt.colorbar(fraction=0.025)

    plt.subplot(1, 3, 3)
    plt.imshow(PSF_3D[2, :, :, Nz//2].abs())
    plt.title('z-axis')
    plt.colorbar(fraction=0.025)

    plt.suptitle('3D PSF : ABS', y=1.01)
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(PSF_3D[0, :, :, Nz//2].angle())
    plt.title('x-axis')
    plt.colorbar(fraction=0.025)

    plt.subplot(1, 3, 2)
    plt.imshow(PSF_3D[1, :, :, Nz//2].angle())
    plt.title('y-axis')
    plt.colorbar(fraction=0.025)

    plt.subplot(1, 3, 3)
    plt.imshow(PSF_3D[2, :, :, Nz//2].angle())
    plt.title('z-axis')
    plt.colorbar(fraction=0.025)

    plt.suptitle('3D PSF : ANGLE', y=1.01)
    plt.show()


# Showing planes of a 3D Tensor
def show_planes(outs, title, N_z=256):  # outs.shape: [Nz, Nx, Ny]
    z_planes = range(0, N_z, max(N_z//8, 1))
    plt.figure(figsize=(20, 2))
    for i, z_idx in enumerate(z_planes):
        plt.subplot(1, len(z_planes), i+1)
        plt.imshow(outs[z_idx], vmin=outs.min(), vmax=outs.max())
        plt.axis('off')
        plt.title(f'z: {z_idx}')

    plt.subplots_adjust(top=0.73)
    plt.suptitle(f'{title} ')
    plt.show()

# Showing specified planes of a 3D Tensor


def show_planes_z(outs, title, z_planes):  # outs.shape: [Nz, Nx, Ny]
    plt.figure(figsize=(20, 2))
    for i, z_idx in enumerate(z_planes):
        plt.subplot(1, len(z_planes), i+1)
        plt.imshow(outs[z_idx], vmin=outs.min(), vmax=outs.max())
        plt.axis('off')
        plt.title(f'z: {z_idx}')

    plt.subplots_adjust(top=0.73)
    plt.suptitle(f'{title}')
    plt.show()


# Showing 2D Images
def show_image(image, title='', fig_size=(5, 5)):
    plt.figure(figsize=fig_size)
    plt.imshow(image, vmax=image.max(), vmin=image.min())
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
