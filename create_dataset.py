import torch
from tqdm import tqdm
import pandas as pd
from libs.forward_lib.physical_model import PhysicalModel
from libs.forward_lib.linearized_process import LinearizedModel
from libs.forward_lib.simulate_data import MnistSimulator


def store_psf():
    nx, ny, nz = 2048, 2048, 512
    
    par_list  = [[1.5, 1.2], [1., .8], [1.3, 1.]]

    for it, param in enumerate(par_list):
        [PhysicalModel.r_index, PhysicalModel.NA] = param
        PM = PhysicalModel(nx, ny, nz, n_patterns=1, device = 'cpu')
        data_to_save = {
            "r_index"   : PM.r_index,
            "NA"        : PM.NA,
            "dimension" : [PM.nx, PM.ny, PM.nz],
            "voxel"     : [PM.dx, PM.dy, PM.dz],
            "matrix"    : PM.exPSF_3D
        }
        path_to_save = f'./data/matrices/field/PSF_{it}.pt'
        torch.save(data_to_save, path_to_save)
    
        log_data = {
            "it" : it,
            "r_index" : PM.r_index,
            "NA" : PM.NA,
            "nx" : PM.nx, "ny" : PM.ny, "nz" : PM.nz,
            "dx" : PM.dx, "dy" : PM.dy, "dz" : PM.dz
        }
        log_path = "./data/matrices/log/PSF.csv"
        new_df = pd.DataFrame(log_data, index = [0])
        new_df.to_csv(log_path, mode='a', header=False, index=False)


def create_A(IT=11):
    nx, ny, nz = 256, 256, 256
    n_patterns = 64
    dd_factor = 16
    LinearizedModel.device = 'cpu'

    # initialize A and store A_r
    LM = LinearizedModel(nx,ny,nz,n_patterns,dd_factor,init_call=True)
    LM.approximate_A()
    LM.save_matrix(it = IT, original_ = True)
    LM.save_matrix(it = IT, original_ = False)
    print(LM)


def dataset_creater(IT = 11):
    LM = LinearizedModel()
    LM.load_matrix(it = IT, original_=True)
    print(LM)

    device = LM.device
    nx, ny, nz = LM.nx, LM.ny, LM.nz
    MS = MnistSimulator(nx, ny, nz, up_factor = 1)  
    num_data = 8

    X_r, Y = torch.tensor([]).to(device), torch.tensor([]).to(device)

    for i_z in tqdm(range(num_data), desc = "Data Point: "):
        MS.update_data()
        MS.reduce_dimension()
        y = (LM.A @ MS.X.flatten()).flatten()
        x_r = MS.X_r.flatten()

        X_r, Y = torch.cat([X_r, x_r.unsqueeze(0)], dim=0), torch.cat([Y,y.unsqueeze(0) ])

    torch.save(X_r,"./data/dataset/X_r.pt")
    torch.save(Y,"./data/dataset/Y.pt")



    









