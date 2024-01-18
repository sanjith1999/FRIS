import torch
import torch.optim as optim


def reconstruct_image_basic(A, y):
    A_pseudo_inv = torch.linalg.pinv(A)   # torch.linalg.pinv(A.T@A)@A.T
    return torch.matmul(A_pseudo_inv, y)


# LU Decomposition
def reconstruct_image_lu(A, y):
    ATA = (A.T@A).float()
    b = (A.T@y).view(-1,1)
    LU, pivots = torch.lu(ATA)
    return torch.lu_solve(b, LU, pivots)

# Cholesky Decomposition
def reconstruct_image_ch(A, y):
    ATA = (A.T@A).float()
    b = (A.T@y).view(-1,1)
    U = torch.cholesky(ATA, upper=False)
    return torch.cholesky_solve(b, U, upper=False)



def reconstruct_image_gd(A, y,Nx,Ny,Nz, device, learning_rate = 1e-1, num_iter = 50000):
    # RANDOM INTIALIZATION OF X
    x = torch.autograd.Variable(torch.rand(Nx*Ny*Nz,1,dtype = torch.float32).to(device), requires_grad=True)

    # Define the optimizer
    optimizer = optim.SGD([x], lr=learning_rate)

    # Gradient descent optimization loop
    for iter in range(num_iter+1):
        loss = torch.norm(A@x-y,p=2)
        loss.backward()
        optimizer.step()
        if(iter%10000 == 0):
            print(f"At Iteration {iter} Error is: {loss}")
        optimizer.zero_grad()
    return x