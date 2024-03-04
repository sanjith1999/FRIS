import torch
import pandas as pd
from tqdm import tqdm
from libs.forward_lib.physical_model import PhysicalModel, psf_model
from libs.forward_lib.linearized_process import LinearizedModel
from libs.forward_lib.simulate_data import MnistSimulator
from libs.forward_lib.read_data import ReadData


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
    nx, ny, nz = 32, 32, 8
    dx, dy, dz = 1., 1., 1.
    NA = .8
    r_index = 1
    IT = 51
    
    PSF = psf_model(NA=NA, Rindex=r_index, lambda_=PhysicalModel.lambda_ ,dx=dx, dy=dy, dz=dz, Nx=nx, Ny=ny, Nz=nz)
    data_to_save = {
        "r_index"   : r_index,
        "NA"        : NA,
        "dimension" : [nx, ny, nz],
        "voxel"     : [dx, dy, dz],
        "matrix"    : PSF
    }
    path_to_save = f'./data/matrices/field/PSF_{IT}.pt'
    torch.save(data_to_save, path_to_save)


# Calculation of matrix A
def create_A():
    nx, ny, nz = 128, 128, 32
    n_patterns = 2
    dd_factor = 8
    PSF_IT = 50
    PhysicalModel.dx, PhysicalModel.dy, PhysicalModel.dz = .25, .25, .25
    PhysicalModel.ep_dx, PhysicalModel.ep_dy = 2., 2.

    LM = LinearizedModel(nx,ny,nz,n_patterns,dd_factor)
    for IT in range(8, 16):
        print(f"\n\nITERATION: {IT+1}\n-------------\n")
        LM.nx, LM.ny, LM.nz, LM.n_patterns, LM.dd_factor = nx, ny, nz, n_patterns, dd_factor
        LM.dx, LM.dy, LM.dz = PhysicalModel.dx, PhysicalModel.dy, PhysicalModel.dz
        LM.init_models(PSF_IT)
        LM.PM.dmd.initialize_patterns(IT)
        LM.PM.dmd.visualize_patterns()
        print(LM)
        LM.find_transformation()
        LM.save_matrix(it = IT)
        LM.prepare_approximate(4,4,4)
        print("\n", LM)
        LM.approximate_transformation()
        LM.save_matrix(it = IT, is_original=False)




# Approximation of A to form a smaller matrix
def approximate_A():
    nx, ny, nz = 32, 32, 8
    n_patterns = 2
    dd_factor = 2
    PSF_IT = 51
    PhysicalModel.dx, PhysicalModel.dy, PhysicalModel.dz = 1., 1., 1.
    PhysicalModel.ep_dx, PhysicalModel.ep_dy = 2., 2.

    LM = LinearizedModel(nx,ny,nz,n_patterns,dd_factor)
    LM.init_models(PSF_IT)
    print(LM)
    for IT in range(16):
        print(f"\n\nITERATION: {IT+1}\n-------------\n")
        LM.PM.dmd.recover_patterns(IT)
        LM.PM.dmd.visualize_patterns()
        LM.find_transformation()
        LM.save_matrix(it = IT, is_original=False)


# Stack up the individually calculated A to form the larger matrix
def stack_up_A(DT = 130):
    n_p = 32
    LM = LinearizedModel()
    stacked_A = torch.tensor([]).to(LM.device)
    for IT in range(int(n_p/2)):
        LM.load_matrix(IT, is_original=False)
        stacked_A = torch.cat((stacked_A, LM.A))
    LM.A = stacked_A
    LM.n_patterns = n_p
    LM.save_matrix(DT, is_original=False)


# Create and store objects
def create_data(IT = 0, batch_size = 2, object_type = "MNIST"):
    nx, ny, nz = 128, 128, 32
    device = 'cuda'
    X_r, X = torch.tensor([]).to(device), torch.tensor([]).to(device)
    if object_type == "MNIST":
        F_MS = MnistSimulator(nx, ny, nz, up_factor=(1,4,4))

        for b in tqdm(range(batch_size), desc = f"Data Point {IT:03}: "):
            F_MS.update_data()
            F_MS.reduce_dimension()
            x, x_r  = F_MS.X.flatten(),  F_MS.X_r.flatten()

            X_r, X = torch.cat([X_r, x_r.unsqueeze(0)], dim=0), torch.cat([X,x.unsqueeze(0) ])
    
    if object_type == "BLOOD_CELL":
        RD = ReadData(nx, ny, nz)
        for b in tqdm(range(batch_size), desc=f"Data Point {IT:03}: "):
            RD.load_object('blood_cell')
            RD.reduce_dimension()
            x, x_r = RD.X.flatten(), RD.X_r.flatten()

            X_r, X = torch.cat([X_r, x_r.unsqueeze(0)], dim=0), torch.cat([X,x.unsqueeze(0) ])
    
    if object_type == "NEURAL_CELL":
        RD = ReadData(nx, ny, nz)
        for b in tqdm(range(batch_size), desc=f"Data Point {IT:03}: "):
            RD.load_object('neural_cell')
            RD.reduce_dimension()
            x, x_r = RD.X.flatten(), RD.X_r.flatten()

            X_r, X = torch.cat([X_r, x_r.unsqueeze(0)], dim=0), torch.cat([X,x.unsqueeze(0) ])
    torch.save(X_r,f"./data/dataset/object/X_r_{IT}.pt")
    torch.save(X,f"./data/dataset/h_object/X_{IT}.pt")


# Calculate Measurement
def run_process():
    for m_it in range(16):
        LM = LinearizedModel()
        LM.load_matrix(m_it)
        for IT in tqdm(range(32), desc = f"Pattern Pair: {m_it+1:02}\t\tData Point: "):
            if m_it ==0:
                X =  torch.load(f"./data/dataset/h_object/X_{IT}.pt").to(LM.device)
                Y = LM.A@X.t()
                torch.save(Y.t(),f"./data/dataset/measurement/Y_{IT}.pt")
            else:
                X =  torch.load(f"./data/dataset/h_object/X_{IT}.pt").to(LM.device)
                Y = torch.load(f"./data/dataset/measurement/Y_{IT}.pt").to(LM.device).t()
                y = LM.A@X.t()
                Y = torch.cat((Y, y))
                torch.save(Y.t(),f"./data/dataset/measurement/Y_{IT}.pt")
        del LM



def combine_data():
    for IT in range(32):
        Yi = torch.load(f"./data/dataset/measurement/Y_{IT}.pt")
        Xri = torch.load(f"./data/dataset/object/X_r_{IT}.pt")
        Xi = torch.load(f"./data/dataset/h_object/X_{IT}.pt")
        if IT ==0:
            S_Y = Yi
            S_Xr = Xri
            S_X = Xi
        else:
            S_Y = torch.cat((S_Y, Yi))
            S_Xr = torch.cat((S_Xr, Xri))
            S_X = torch.cat((S_X, Xi))
    print("Object:", S_Xr.shape,"Measurement:",S_Y.shape)
    torch.save(S_X, "./data/dataset/h_object/S_X.pt")
    torch.save(S_Xr, "./data/dataset/object/S_Xr.pt")
    torch.save(S_Y, "./data/dataset/measurement/S_Y.pt")
        






