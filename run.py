from libs.forward_lib.physical_model import PhysicalModel
import torch
import pandas as pd

def store_psf():
    """ 
     
    """
    nx, ny, nz = 4096, 4096, 1024
    
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


if __name__ == "__main__":
    store_psf()