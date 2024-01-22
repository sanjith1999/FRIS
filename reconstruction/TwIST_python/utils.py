import numpy as np
from scipy.signal import convolve2d
from numpy.fft import fftn

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

def L1Norm_FD(X, nx, ny, nz):
    X_3D = X.reshape((nz, ny, nx))
    X_fourier = fftn(X_3D)
    l1_norm_fourier = np.sum(np.abs(X_fourier))
    return l1_norm_fourier

def soft(x, T):
    y = np.maximum(np.abs(x) - T, 0)
    y = y/(y + T)*x
    return y

