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
    nx, ny, nz = 32, 32, 32
    dx, dy, dz = .32, .32, .32
    NA = .8
    r_index = 1
    IT = 51
    
    PSF = psf_model(NA=NA, Rindex=r_index, lambda_=PhysicalModel.lambda_ ,dx=dx, dy=dy, dz=dz, Nx=nx, Ny=ny, Nz=nz)
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
    PSF_IT = 50
    PhysicalModel.dx, PhysicalModel.dy, PhysicalModel.dz = .08, .08, .08

    LM = LinearizedModel(nx,ny,nz,n_patterns,dd_factor)
    print(LM)
    LM.init_models(PSF_IT)
    for IT in range(2,32):
        print(f"\n\nITERATION: {IT+1}\n-------------\n")
        LM.PM.dmd.initialize_patterns(IT)
        LM.PM.dmd.visualize_patterns()
        LM.find_transformation()
        LM.save_matrix(it = IT)




# Approximation of A to form a smaller matrix
def approximate_A():
    nx, ny, nz = 32, 32, 32
    n_patterns = 2
    dd_factor = 2
    PSF_IT = 51
    PhysicalModel.dx, PhysicalModel.dy, PhysicalModel.dz = .32, .32, .32

    LM = LinearizedModel(nx,ny,nz,n_patterns,dd_factor)
    LM.init_models(PSF_IT)
    print(LM)
    for IT in range(0,16):
        print(f"\n\nITERATION: {IT+1}\n-------------\n")
        LM.PM.dmd.recover_patterns(IT)
        LM.PM.dmd.visualize_patterns()
        LM.find_transformation()
        LM.save_matrix(it = IT, is_original=False)


# Stack up the individually calculated A to form the larger matrix
def stack_up_A(DT = 129):
    LM = LinearizedModel()
    stacked_A = torch.tensor([]).to(LM.device)
    for IT in range(16):
        LM.load_matrix(IT, is_original=False)
        stacked_A = torch.cat((LM.A, stacked_A))
    LM.A = stacked_A
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
    started = False

    X =  torch.load(f"./data/dataset/X_{IT}.pt").to(LinearizedModel.device)

    for m_it in tqdm(range(16), desc = "Pattern Pair: "):
        LM = LinearizedModel()
        LM.load_matrix(m_it)
        if not started:
            started = True
            Y = LM.A@X.t()
        else:
            y = LM.A@X.t()
            Y = torch.cat((Y, y))
        del LM
    
    torch.save(Y.t(),f"./data/dataset/Y_{IT}.pt")












