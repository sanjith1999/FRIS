import torch
import libs.forward_model as fm

# Forward model of a single measurement
def forward_model(X, Ht_2D=None, object_plane = 0):
    global X_masked
    ht_3D = torch.zeros(1, fm.Nz, fm.Nx, fm.Ny).float().to(fm.device)
    ht_3D[:, fm.Nz//2] = Ht_2D
    H1 = fm.conv_3D(fm.exPSF_3D, ht_3D)
    H2 = (H1.abs()**2).sum(dim=0)

    X_masked = torch.zeros_like(X)
    X_masked[0,object_plane, :, :] = X[0, object_plane, :, :]
    H3 = X_masked * H2
    Y = fm.conv_3D(fm.emPSF_3D, H3).abs()[0]
    det_Y = Y[fm.Nz//2, :, :]
    return det_Y


# Extended forward model
def extended_forward_model(X, object_plane = 0, verbose = False):
    Y = torch.tensor([]).to(fm.device)
    Yi_list = []
    for Ht_2D in fm.Ht_2D_list:
        Yi = forward_model(X, Ht_2D, object_plane=object_plane)
        if verbose: 
            Yi_list.append(Yi.detach().cpu())
        Yi_flatten = Yi.flatten()
        Y = torch.cat((Y, Yi_flatten), dim=0)
    if verbose:
        fm.show_images(Yi_list)

    return Y




