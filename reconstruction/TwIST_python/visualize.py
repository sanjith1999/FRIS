import matplotlib.pyplot as plt
import numpy as np


def visualize_A(A, nx, ny, nz, np, m, df, figsize=(8, 8)):
    plt.figure(figsize=figsize)
    plt.imshow(A, cmap='viridis')
    # vertical lines
    for col in range(nx*ny, nx*ny*nz, nx*ny):
        plt.axvline(x=col + 0.5, color='red', linestyle='--', linewidth=1)
    # horizontal lines
    for row in range(int(nx*df*ny*df), int(nx*df*ny*df*m*np), int(nx*df*ny*df)):
        plt.axhline(y=row + 0.5, color='blue', linestyle='--', linewidth=1)
    plt.title('Measurement matrix A')
    plt.show()

def visualize_X(X, nx, ny, nz, figsize=(15, 15), planes_to_plot=None):
    if planes_to_plot is None:
        planes_to_plot = range(nz)
    num_subplots = len(planes_to_plot)

    num_cols = num_subplots # int(np.ceil(num_subplots / 2))
    num_rows = 1            # int(np.ceil(num_subplots / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, plane_idx in enumerate(planes_to_plot):
        plane_data = X[plane_idx*nx*ny : (plane_idx+1)*nx*ny, 0].reshape((ny, nx))
        axes[i].imshow(plane_data, cmap='viridis')
        axes[i].set_title(f'Plane {plane_idx+1}')
    plt.tight_layout()
    plt.show()

def visualize_y(y, nx, ny, df, m, np, figsize=(15, 15)):
    num_cols = m
    num_rows = np

    nx = int(nx*df)
    ny = int(ny*df)
    step = ny*nx

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i in range(m):
        for j in range(np):
            index = i*np*step + j*step
            plane_data = y[index : index+step].reshape((ny, nx))
            axes[i*np + j].imshow(plane_data, cmap='viridis')
            axes[i*np + j].set_title(f'Pattern {i + 1}, Plane # {j + 1}')
    plt.tight_layout() 
    plt.show()

def obj_mse_twist(obj_twist, times_twist, mse_twist, figsize=(15, 15)):
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

def comparison_twist(X, xtwist, nx, ny, nz, figsize=(15, 15), planes_to_plot=None):
    if planes_to_plot is None:
        planes_to_plot = range(nz)
    num_subplots = len(planes_to_plot)*2

    num_cols = len(planes_to_plot) 
    num_rows = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, plane_idx in enumerate(planes_to_plot):
        plane_X = X[plane_idx*nx*ny : (plane_idx+1)*nx*ny, 0].reshape((ny, nx))
        axes[i].imshow(plane_X, cmap='viridis')
        axes[i].set_title(f'Original {plane_idx+1}')
        plane_xtwist = xtwist[plane_idx*nx*ny : (plane_idx+1)*nx*ny, 0].reshape((ny, nx))
        axes[i+num_cols].imshow(plane_xtwist, cmap='viridis')
        axes[i+num_cols].set_title(f'Reconstructed {plane_idx+1}')
    plt.tight_layout()
    plt.show()



