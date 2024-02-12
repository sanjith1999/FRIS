import numpy as np
from scipy.signal import convolve2d
import pywt
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def A(X, FM, nx, ny, nz):
    return FM@np.reshape(X, (nz*ny*nx, 1))
def AT(y, FM, nx, ny, nz):
    return np.reshape(FM.T@y, (nz*ny*nx, 1))


def tvdenoise2D(f, lambda_val, iters, nx, ny, nz):
    if lambda_val < 0:
        raise ValueError("Parameter lambda must be nonnegative.")
    
    lambda_val /= nz

    dt = 0.25
    u3D = np.zeros((nx*ny*nz, 1))

    for j in range(nz):
        fz = f[j*nx*ny: (j+1)*nx*ny, 0].reshape((ny, nx))
        N = fz.shape
        id = np.arange(1, N[0]).tolist() + [N[0] - 1]
        iu = [0] + np.arange(0, N[0] - 1).tolist()
        ir = np.arange(1, N[1]).tolist() + [N[1] - 1]
        il = [0] + np.arange(0, N[1] - 1).tolist()
        p1 = np.zeros_like(fz)
        p2 = np.zeros_like(fz)
        divp = np.zeros_like(fz)

        for _ in range(iters):
            z = divp - fz*lambda_val
            z1 = z[:, ir] - z
            z2 = z[id, :] - z
            denom = 1 + dt*np.sqrt(z1**2 + z2**2)
            p1 = (p1 + dt*z1)/denom
            p2 = (p2 + dt*z2)/denom
            divp = p1 - p1[:, il] + p2 - p2[iu, :]

        uz = fz - divp/lambda_val
        u3D[j*nx*ny: (j+1)*nx*ny, 0] = uz.flatten()

    return u3D


def TVnorm2D(X, nx, ny, nz):
    y = 0
    for j in range(nz):
        xz = X[j*nx*ny : (j+1)*nx*ny, 0]
        xz = np.reshape(xz, (ny, nx))
        kernel_x = np.array([[0, 1, -1]])
        kernel_y = np.array([[0], [1], [-1]])
        diff_h = convolve2d(xz, kernel_x, mode='same', boundary='wrap')
        diff_v = convolve2d(xz, kernel_y, mode='same', boundary='wrap')
        y += np.sum(np.sqrt(diff_h**2 + diff_v**2))
    return y


def L1Norm_DWT(X, nx, ny, nz):
    X = X.reshape((nz, ny, nx))
    coeffs = pywt.wavedecn(X, 'db4', level=1)
    # Access the wavelet coefficients (excluding approximation coefficients)
    coeffs = coeffs[1:]
    flattened_coeffs = np.concatenate([np.array(list(c.values())).flatten() for c in coeffs])
    l1_norm = np.linalg.norm(flattened_coeffs, ord=1)
    return l1_norm

def soft_DWT(X, T, nx, ny, nz):
    X = X.reshape((nz, ny, nx))

    if nz == 1:
        axes = (1,2)
    else:
        axes = (0,1,2)

    coeffs = pywt.dwtn(X, 'db4', mode='symmetric', axes=axes)

    def soft_thresholding(v, T):
        return np.array(np.sign(v)*np.maximum(np.abs(v) - T, 0)) 
    
    thresholded_coeffs = {k: soft_thresholding(v, T) for k,v in coeffs.items()}
    if nz == 1:
        thresholded_coeffs['aa'] = coeffs['aa']
    else:
        thresholded_coeffs['aaa'] = coeffs['aaa']
    reconstructed_image = pywt.idwtn(thresholded_coeffs, 'db4', mode='symmetric', axes=axes)
    return reconstructed_image.reshape(nx*ny*nz, 1)

def soft(x, T):
    y = np.maximum(np.abs(x) - T, 0)
    y = y/(y + T)*x
    return y


def plot_obj_mse(obj_twist, times_twist, mse_twist, figsize=(15, 15)):
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.semilogy(times_twist, obj_twist, 'r', linewidth=2)
    plt.ylabel('Obj. function')
    plt.xlabel('CPU time (sec)')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(times_twist[1:], mse_twist[1:], 'r', linewidth=2)
    plt.ylabel('MSE')
    plt.xlabel('CPU time (sec)')
    plt.show()

    print(f'TwIST CPU time: {times_twist[-1]}')
    print(f'MSE Loss: {mse_twist[-1]:.4e}')


def comparison2(X, xhat, nx, ny, nz, figsize=(15, 15), planes_to_plot=None):
    if planes_to_plot is None:
        planes_to_plot = range(nz)
    num_subplots = len(planes_to_plot)*2

    num_cols = len(planes_to_plot) 
    num_rows = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_subplots == 1:
        axes = [axes]
    axes = axes.flatten()
    
    for i, plane_idx in enumerate(planes_to_plot):
        plane_X = X[plane_idx*nx*ny : (plane_idx+1)*nx*ny, 0].reshape((ny, nx))
        axes[i].imshow(plane_X, cmap='viridis')
        axes[i].set_title(f'P {plane_idx+1}')

        plane_xhat = xhat[plane_idx*nx*ny : (plane_idx+1)*nx*ny, 0].reshape((ny, nx))
        axes[i+num_cols].imshow(plane_xhat, cmap='viridis')

    ssim_values = []
    psnr_values = []
    for i in range(nz):
        plane_X = X[i*nx*ny : (i+1)*nx*ny, 0].reshape((ny, nx))
        plane_xhat = xhat[i*nx*ny : (i+1)*nx*ny, 0].reshape((ny, nx))

        ssim_plane = ssim(plane_X, plane_xhat, data_range=np.max(plane_xhat)-np.min(plane_xhat))
        ssim_values.append(ssim_plane) 

        psnr_plane = psnr(plane_X, plane_xhat, data_range=np.max(plane_xhat)-np.min(plane_xhat))
        psnr_values.append(psnr_plane)

    plt.tight_layout()
    plt.show()

    for i in range(nz):
        print("plane {} | SSIM = {:.4f} | PSNR = {:.4f} dB".format(i+1, ssim_values[i], psnr_values[i]))

    return ssim_values, psnr_values



