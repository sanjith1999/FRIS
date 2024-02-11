import torch
from tqdm import tqdm
import numpy as np
from libs.forward_lib.physical_model import PhysicalModel
from libs.forward_lib.visualizer import show_image

class LinearizedModel:
    """ 
    Class: linearize the whole forward process into a matrix, approximate the matrix with a low dimensional version
    """

    # Class Variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dx, dy, dz = PhysicalModel.dx, PhysicalModel.dy, PhysicalModel.dz
    v_dx, v_dy, v_dz = .1, .1, .1                                               #um
    v_nx, v_ny, v_nz = int(v_dx/dx), int(v_dy/dy), int(v_dz/dz)

    def __init__(self, nx, ny, nz,n_patterns ,dd_factor = 1, n_planes=1, init_call = False):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dd_factor = dd_factor
        self.n_planes = n_planes
        self.n_patterns = n_patterns
        self.calculate_dimensions()
        if init_call:
            self.init_models()

    def __str__(self):
        desc = ""
        desc += "Linearized Model Specifications\n----------------------------------------------\n"
        desc += f"NA \t\t\t\t: {PhysicalModel.NA}\n"
        desc += f"Space Dimension \t\t: {self.nx*self.dx:.3f}um × {self.ny*self.dy:.3f}um × {self.nz*self.dz:.3f}um\n"
        desc += f"Analog Voxel Size \t\t: {self.dx}um × {self.dy}um × {self.dz}um\n"
        desc += f"Reduced Voxel Size \t\t: {self.v_dx}um × {self.v_dy}um × {self.v_dz}um\n"
        desc += f"Original Shape \t\t\t: {self.nx} × {self.ny} × {self.nz}\n"
        desc += f"Reduced Shape \t\t\t: {self.r_nx} × {self.r_ny} × {self.r_nz}\n"
        desc += f"DMD Patterns \t\t\t: {self.n_patterns}\n"
        desc += f"Measurement Plane\t\t: {self.n_planes}\n"
        desc += f"Detector Pool size \t\t: {self.dd_factor}×{self.dd_factor}\n"
        desc += f"Computational Device \t\t: {self.device}"
        return desc
    

    def init_models(self):    
        """ 
        Method: initializing the physical model with necessary parameters & initialize the matrix A with zeros
        """
        self.PM = PhysicalModel(self.nx, self.ny, self.nz,self.n_patterns, self.dd_factor, self.n_planes, self.device)
        self.A = torch.zeros(int(self.nx/self.dd_factor)*int(self.ny/self.dd_factor)*self.n_planes*self.n_patterns, self.nx*self.ny*self.nz).float().to(self.device)
        self.find_transformation()

    
    def find_transformation(self):
        """ 
        Method: calculation of A with the help of impulses in extended X
        """
        I = torch.zeros(1, self.nz, self.nx, self.ny).float().to(self.device)
        for i_z in tqdm(range(self.nz), desc = "Plane Calculations: "):
            for i_x in range(self.nx):
                for i_y in range(self.ny):
                    I[0, i_z, i_x, i_y] = 1
                    self.A[:, i_z*self.ny*self.nx + i_x*self.ny+i_y] = self.PM.extended_propagation(I)
                    I[0, i_z, i_x, i_y] = 0
        return "SUCCESS...!"
    

    def calculate_dimensions(self):
        self.v_nx, self.v_ny, self.v_nz = max(1, int(self.v_dx/self.dx)), max(int(self.v_dy/self.dy),1), max(1,int(self.v_dz/self.dz))
        self.r_nx, self.r_ny, self.r_nz = int(self.nx/self.v_nx), int(self.ny/self.v_ny), int(self.nz/self.v_nz)

    def approximate_A(self):
        """ 
        Method: find an approximate to A
        """
        tr_A = self.find_reducer_matrix()
        self.A_r = self.A @ tr_A
        print("A approximation successful...!")

    def find_reducer_matrix(self):
        """ 
        Method: find the matrix transformation required to go from A -> A_r
        """
        tr_A = torch.zeros(self.nx*self.ny*self.nz, self.r_nx * self.r_ny* self.r_nz).float().to(self.device)

        I = torch.zeros(1, self.nz, self.nx, self.ny).float().to(self.device)
        for i_z in tqdm(range(self.r_nz), desc = "Reduced Plane Calculations: "):
            for i_x in range(self.r_nx):
                for i_y in range(self.r_ny):
                    I[0, i_z:i_z+self.v_nz, i_x:i_x+self.v_nx, i_y:i_y+self.v_ny] = 1
                    cur_u = I.flatten()
                    tr_A[:, i_z*self.r_ny*self.r_nx + i_x*self.r_ny+i_y] = cur_u
                    I[0, i_z:i_z+self.v_nz, i_x:i_x+self.v_nx, i_y:i_y+self.v_ny] = 0

        return tr_A
    
    def save_original_matrix(self,it = 0, original_ = True):
        """ 
        Method: function to save matrix A reduced/original
        """
        if original_:
            path = f"./data/matrices/original/A_{it}.pt" 
        else:
            path = f"./data/matrices/reduced/A_{it}.pt" 
        data_to_save = {
            "NA"            :   PhysicalModel.NA,
            "voxel_size"    :   [self.dx, self.dy, self.dz],
            "dimensions"    :   [self.nx, self.ny, self.nz],
            "p_dimensions"  :   [PhysicalModel.ep_dx, PhysicalModel.ep_dy],
            "c_patterns"    :   self.n_patterns,
            "c_planes"      :   self.n_planes,
            "down_factor"   :   self.dd_factor,
            "matrix"        :   self.A
        }
        torch.save(data_to_save, path)

    def load_matrix(self, it = 0, original_ = True):
        """ 
        Method: function to load an available matrix together with parameters
        """
        if original_:
            path = f"./data/matrices/original/A_{it}.pt" 
        else:
            path = f"./data/matrices/reduced/A_{it}.pt" 
        loaded_data = torch.load(path)

        PhysicalModel.NA                            = loaded_data["NA"]
        [self.dx, self.dy, self.dz]                 = loaded_data["voxel_size"]
        [self.nx, self.ny, self.nz]                 = loaded_data["dimensions"]
        [PhysicalModel.ep_dx, PhysicalModel.ep_dy]  = loaded_data["p_dimensions"]
        self.n_patterns                             = loaded_data["c_patterns"]
        self.n_planes                               = loaded_data["c_planes"]
        self.dd_factor                              = loaded_data["down_factor"]
        self.calculate_dimensions()
        if original_:
            self.A                                  = loaded_data["matrix"]
            try:
                aux_data = torch.load(f"./data/matrices/reduced/A_{it}.pt")
                self.A_r = aux_data["matrix"]
            except:
                self.approximate_A()
                # self.save_reduced_matrix(it)
        else:
            self.A_r                                = loaded_data["matrix"]
            # try:
            #     aux_data = torch.load(f"./data/matrices/original/A_{it}.pt")
            #     self.A = aux_data["matrix"]
            # except:
            #     print("A is Invalid...!")

    def visualize_A(self, original_ = True):
        """ 
        Method: function to visualize the matrices
        """
        if original_:
            A_ = self.A.detach().cpu().numpy()
        else:
            A_ = self.A_r.detach().cpu().numpy()
        A_[A_==0] = 1e-30
        A_ = np.log(A_)
        show_image(A_,fig_size=(12,8),title=r"A(Log Scale)")