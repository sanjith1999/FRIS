import torch
import pandas as pd
from tqdm import tqdm
from libs.forward_lib.physical_model import PhysicalModel, psf_model
from libs.forward_lib.linearized_process import LinearizedModel
from libs.forward_lib.simulate_data import MnistSimulator


def field_related_calculations():
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


# Calculate PSF of necessary Dimension and Store it before finding the Transformation A
def store_PSF():
    nx, ny, nz = 128, 128, 128
    NA = .8
    r_index = 1
    IT = 51                                                         # Let me stick to numbers between 50-100 here
    
    PSF = psf_model(NA=NA, Rindex=r_index, lambda_=PhysicalModel.lambda_ ,dx=PhysicalModel.dx, dy=PhysicalModel.dy, dz=PhysicalModel.dz, Nx=nx, Ny=ny, Nz=nz)
    data_to_save = {
        "r_index"   : r_index,
        "NA"        : NA,
        "dimension" : [nx, ny, nz],
        "voxel"     : [PhysicalModel.dx, PhysicalModel.dy, PhysicalModel.dz],
        "matrix"    : PSF
    }
    path_to_save = f'./data/matrices/field/PSF_{IT}.pt'
    torch.save(data_to_save, path_to_save)


# Calculation of matrix A
def create_A():
    nx, ny, nz = 128, 128, 128
    n_patterns = 2
    dd_factor = 8
    # PhysicalModel.device = 'cpu'

    LM = LinearizedModel(nx,ny,nz,n_patterns,dd_factor)
    print(LM)
    for IT in range(16, 32):
        LM.PM.init_dmd()
        LM.PM.dmd.visualize_patterns()
        LM.find_transformation()
        LM.save_matrix(it = IT)




# Approximation of A to form a smaller matrix
def approximate_A():
    for IT in range(0,16):
        LM = LinearizedModel()
        LM.load_matrix(it=IT, is_original=True)
        print(f"A: {IT}")
        LM.approximate_A()
        LM.save_matrix(IT, is_original=False)
        del LM


# Stack up the individually calculated A to form the larger matrix
def stack_up_A(DT = 129):
    LM = LinearizedModel()
    stacked_A = torch.tensor([]).to(LM.device)
    for IT in range(16):
        LM.load_matrix(IT, is_original=False)
        stacked_A = torch.cat((LM.A_r, stacked_A))
    LM.A_r = stacked_A
    LM.save_matrix(DT, is_original=False)


# Create and store objects
def create_data(IT = 0, batch_size = 2):
    nx, ny, nz = 128, 128, 128
    device = 'cuda'
    MS = MnistSimulator(nx, ny, nz, up_factor = 4)  

    X_r, X = torch.tensor([]).to(device), torch.tensor([]).to(device)

    for b in tqdm(range(batch_size), desc = "Data Point: "):
        MS.update_data()
        MS.reduce_dimension()
        x, x_r  = MS.X.flatten(),  MS.X_r.flatten()

        X_r, X = torch.cat([X_r, x_r.unsqueeze(0)], dim=0), torch.cat([X,x.unsqueeze(0) ])
    

    torch.save(X_r,f"./data/dataset/X_r_{IT}.pt")
    torch.save(X,f"./data/dataset/X_{IT}.pt")


# Calculate Measurement
def run_process(IT = 0):
    described = False

    X =  torch.load(f"./data/dataset/X_{IT}.pt").to(LinearizedModel.device)

    for m_it in tqdm(range(16), desc = "Pattern Pair: "):
        LM = LinearizedModel()
        LM.load_matrix(m_it)
        if not described:
            described = True
            Y = LM.A@X.t()
        else:
            y = LM.A@X.t()
            Y = torch.cat((Y, y))
        del LM
    
    torch.save(Y.t(),f"./data/dataset/Y_{IT}.pt")












