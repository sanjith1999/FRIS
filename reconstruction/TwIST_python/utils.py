import numpy as np
from scipy.signal import convolve2d

def A(X, FM, nx, ny, nz):
    return FM@np.reshape(X, (nz*ny*nx, 1))
def AT(y, FM, nx, ny, nz):
    return np.reshape(FM.T@y, (nz*ny*nx, 1))


def tvdenoise2D(f, lambda_val, iters, nx, ny, nz, tol=1e-2):
    if lambda_val < 0:
        raise ValueError("Parameter lambda must be nonnegative.")

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
        lastdivp = np.ones_like(fz)

        for _ in range(iters):
            lastdivp = divp.copy()
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

def wraparound(x, m):
    mx, nx = x.shape
    mm, nm = m.shape

    if mm > mx or nm > nx:
        raise ValueError('Mask does not fit inside array')

    mo, no = (1 + mm) // 2, (1 + nm) // 2  # reflected mask origin
    ml, nl = mo - 1, no - 1  # mask left/above origin
    mr, nr = mm - mo, nm - no  # mask right/below origin
    me, ne = mx - ml + 1, nx - nl + 1  # reflected margin in input
    mt, nt = mx + ml, nx + nl  # top of image in output
    my, ny = mx + mm - 1, nx + nm - 1  # output size

    y = np.zeros((my, ny))
    y[mo:mt, no:nt] = x  # central region

    if ml > 0:
        y[0:ml, no:nt] = x[me:mx, :]  # top side
        if nl > 0:
            y[0:ml, 0:nl] = x[me:mx, ne:nx]  # top left corner
        if nr > 0:
            y[0:ml, nt:ny] = x[me:mx, 0:nr]  # top right corner
    if mr > 0:
        y[mt:my, no:nt] = x[0:mr, :]  # bottom side
        if nl > 0:
            y[mt:my, 0:nl] = x[0:mr, ne:nx]  # bottom left corner
        if nr > 0:
            y[mt:my, nt:ny] = x[0:mr, 0:nr]  # bottom right corner
    if nl > 0:
        y[mo:mt, 0:nl] = x[:, ne:nx]  # left side
    if nr > 0:
        y[mo:mt, nt:ny] = x[:, 0:nr]  # right side

    return y

def conv2c(x, h):
    # x = wraparound(x, h)
    y = convolve2d(x, h, mode='same', boundary='wrap')
    return y

def diffh(x):
    h = np.array([0, 1, -1])
    h = h.reshape((1, -1))
    sol = conv2c(x, h)
    return sol

def diffv(x):
    h = np.array([[0], [1], [-1]])
    sol = conv2c(x, h) 
    return sol

def TVnorm2D(X, nx, ny, nz):
    y = 0
    for j in range(nz):
        xz = X[j*nx*ny : (j+1)*nx*ny, 0]
        xz = np.reshape(xz, (ny, nx))
        y += np.sum(np.sqrt(diffh(xz)**2 + diffv(xz)**2))
    return y

def soft(x, T):
    y = np.maximum(np.abs(x) - T, 0)
    y = y/(y + T)*x
    return y

