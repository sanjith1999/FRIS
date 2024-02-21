import torch
from tqdm import tqdm
from libs.efficient_lib.efficient_model import EfficientModel
from libs.forward_lib.visualizer import show_image
import pandas as pd

class EfficientProcess:
    """ 
    Class: linearize the whole forward process into a matrix, approximate the matrix with a low dimensional version
    """

    # Class Variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dx, dy, dz = EfficientModel.dx, EfficientModel.dy, EfficientModel.dz

    def __init__(self, nx = 16, ny=16, nz=16,n_patterns=2,dd_factor = 1, n_planes=1):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dd_factor = dd_factor
        self.n_planes = n_planes
        self.n_patterns = n_patterns
        self.measure_pp = int(nx/self.dd_factor)*int(ny/self.dd_factor)

    def __str__(self):
        desc = ""
        desc += "Linearized Model Specifications\n----------------------------------------------\n"
        desc += f"NA \t\t\t\t: {EfficientModel.NA}\n"
        desc += f"Space Dimension \t\t: {self.nx*self.dx:.3f}um × {self.ny*self.dy:.3f}um × {self.nz*self.dz:.3f}um\n"
        desc += f"Analog Voxel Size \t\t: {self.dx}um × {self.dy}um × {self.dz}um\n"
        desc += f"Original Shape \t\t\t: {self.nx} × {self.ny} × {self.nz}\n"
        desc += f"DMD Patterns \t\t\t: {self.n_patterns}\n"
        desc += f"Measurement Plane\t\t: {self.n_planes}\n"
        desc += f"Detector Pool size \t\t: {self.dd_factor}×{self.dd_factor}\n"
        desc += f"Computational Device \t\t: {self.device}\n\n"
        return desc
    

    def init_models(self):    
        """ 
        Method: initializing the physical model with necessary parameters & initialize the matrix A with zeros
        """
        self.EM = EfficientModel(self.nx, self.ny, self.nz,self.n_patterns, self.dd_factor, self.n_planes, self.device)
        self.A = torch.zeros(int(self.nx/self.dd_factor)*int(self.ny/self.dd_factor)*self.n_planes*self.n_patterns, self.nx*self.ny*self.nz).float().to(self.device)
        self.find_transformation()

    def find_transformation(self):
        """ 
        Method: calculation of A with the help of impulses in extended X
        """
        for i_p in range(self.n_patterns):
           self.EM.propagate_dmd(p_no = i_p+1)
           for i_z in range(self.nz):
                for i_x in tqdm(range(self.nx), desc = f"Pattern: {i_p+1}/{self.n_patterns}\t Nz: {i_z+1}/{self.nz}\t Nx: "):
                    for i_y in range(self.ny):
                        self.A[i_p*self.measure_pp:i_p*self.measure_pp+ self.measure_pp, i_z*self.ny*self.nx + i_x*self.ny+i_y] = self.EM.propagate_object((i_x, i_y, i_z)).flatten()
        return "SUCCESS...!"
    

    
    def save_matrix(self,it = 100):
        """ 
        Method: function to save matrix A reduced/original
        """
        path = f"./data/matrices/original/A_{it}.pt" 
        data_to_save = {
            "NA"            :   EfficientModel.NA,
            "voxel_size"    :   [self.dx, self.dy, self.dz],
            "dimensions"    :   [self.nx, self.ny, self.nz],
            "p_dimensions"  :   [EfficientModel.ep_dx, EfficientModel.ep_dy],
            "c_patterns"    :   self.n_patterns,
            "c_planes"      :   self.n_planes,
            "down_factor"   :   self.dd_factor,
            "matrix"        :   self.A
        }
        torch.save(data_to_save, path)
        self.log_matrix(it)

    def log_matrix(self, it = 100):
        logpath =  f"./data/matrices/log/A_original.csv" 
        data_to_save = {
            "it": it, "NA"   : EfficientModel.NA, "dx" :   self.dx, "dy" : self.dy, "dz" : self.dz, "nx" : self.nx, "ny" : self.ny, "nz" : self.nz,
            "E_dx" : EfficientModel.ep_dx, "E_dy" : EfficientModel.ep_dy, "n_patterns" : self.n_patterns, "n_planes" : self.n_planes,
            "down_factor" : self.dd_factor
        }
        new_df = pd.DataFrame(data_to_save, index = [0])
        new_df.to_csv(logpath, mode='a', header=False, index=False)