import torch
from tqdm import tqdm
import numpy as np
from libs.forward_lib.physical_model import PhysicalModel
from libs.forward_lib.visualizer import show_image
import pandas as pd


# noinspection PyAttributeOutsideInit
class LinearizedModel:
    """ 
    Class: linearize the whole forward process into a matrix, approximate the matrix with a low dimensional version
    Note: before finding transformation manually init_model ~ PSF, init_DMD pattern ~ creating patterns
    """

    # Class Variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, nx=16, ny=16, nz=16, n_patterns=2, dd_factor=1, n_planes=1):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dd_factor = dd_factor
        self.n_planes = n_planes
        self.n_patterns = n_patterns
        self.measure_pp = int(nx / self.dd_factor) * int(ny / self.dd_factor)
        self.dx, self.dy, self.dz = PhysicalModel.dx, PhysicalModel.dy, PhysicalModel.dz
        self.ep_dx, self.ep_dy = PhysicalModel.ep_dx, PhysicalModel.ep_dy

    def __str__(self):
        desc = ""
        desc += "Linearized Model Specifications\n----------------------------------------------\n"
        desc += f"NA \t\t\t\t: {PhysicalModel.NA}\n"
        desc += f"Space Dimension \t\t: {self.nx * self.dx:.3f}um × {self.ny * self.dy:.3f}um × {self.nz * self.dz:.3f}um\n"
        desc += f"Voxel Size \t\t\t: {self.dx}um × {self.dy}um × {self.dz}um\n"
        desc += f"Original Shape \t\t\t: {self.nx} × {self.ny} × {self.nz}\n"
        desc += f"DMD Patch Size \t\t\t: {self.ep_dx}um × {self.ep_dy}um\n"
        desc += f"DMD Patterns \t\t\t: {self.n_patterns}\n"
        desc += f"# of Plane\t\t\t: {self.n_planes}\n"
        desc += f"Detector Pool size \t\t: {self.dd_factor}×{self.dd_factor}\n"
        desc += f"Computational Device \t\t: {self.device}\n\n"
        return desc

    def init_models(self, LOAD_IT=-1):
        self.PM = PhysicalModel(self.nx, self.ny, self.nz, self.n_patterns, self.dd_factor, self.n_planes, self.device)
        if LOAD_IT >= 0:
            self.PM.load_psf(LOAD_IT)
        else:
            self.PM.init_psf()
        self.A = torch.zeros(int(self.nx / self.dd_factor) * int(self.ny / self.dd_factor) * self.n_planes * self.n_patterns, self.nx * self.ny * self.nz).float().to(self.device)

    def find_transformation(self):
        """
        Method: calculation of A with the help of impulses in X
        """
        for i_p in range(self.n_patterns):
            self.PM.propagate_dmd(p_no=i_p + 1)
            for i_z in tqdm(range(self.nz), desc=f"Pattern: {i_p + 1}/{self.n_patterns}\t Nz: "):
                for i_x in range(self.nx):
                    for i_y in range(self.ny):
                        self.A[i_p * self.measure_pp:i_p * self.measure_pp + self.measure_pp, i_z * self.ny * self.nx + i_x * self.ny + i_y] = self.PM.propagate_impulse((i_x, i_y, i_z)).flatten()
        return "SUCCESS...!"

    def prepare_approximate(self, v_nx, v_ny, v_nz):
        """
        Method: prepare the model for approximation,
        """
        self.v_nx, self.v_ny, self.v_nz = v_nx, v_ny, v_nz
        self.dx, self.dy, self.dz = self.dx * v_nx, self.dy * v_ny, self.dz * v_nz
        self.nx, self.ny, self.nz = int(self.nx / v_nx), int(self.ny / v_ny), int(self.nz / v_nz)
        self.dd_factor = int(self.dd_factor / v_nx)
        self.A = torch.zeros(int(self.nx / self.dd_factor) * int(self.ny / self.dd_factor) * self.n_planes * self.n_patterns, self.nx * self.ny * self.nz).float().to(self.device)

    def approximate_transformation(self):
        """
        Method: approximation of A via propagating patch impulses,
        """
        Ip = torch.zeros(1, self.nz * self.v_nz, self.nx * self.v_nx, self.ny * self.v_ny).to(self.device)
        for i_p in range(self.n_patterns):
            self.PM.propagate_dmd(p_no=i_p + 1)
            for i_z in tqdm(range(self.nz), desc=f"A-Pattern: {i_p + 1}/{self.n_patterns}\t Nz: "):
                for i_x in range(self.nx):
                    for i_y in range(self.ny):
                        Ip[0, i_z * self.v_nz:(i_z + 1) * self.v_nz, i_x * self.v_nx:(i_x + 1) * self.v_nx, i_y * self.v_ny:(i_y + 1) * self.v_ny] = 1
                        self.A[i_p * self.measure_pp:(i_p + 1) * self.measure_pp, i_z * self.ny * self.nx + i_x * self.ny + i_y] = self.PM.propagate_patch(Ip).flatten()
                        Ip[0, i_z * self.v_nz:(i_z + 1) * self.v_nz, i_x * self.v_nx:(i_x + 1) * self.v_nx, i_y * self.v_ny:(i_y + 1) * self.v_ny] = 0

    def save_matrix(self, it=100, is_original=True, is_approx = False):
        """ 
        Method: function to save matrix A reduced/original
        """
        path = f"./data/matrices/original/A_{it}.pt" if is_original else (f"./data/matrices/approximated/A_{it}.pt" if is_approx else f"./data/matrices/reduced/A_{it}.pt")
        data_to_save = {"NA": PhysicalModel.NA, "voxel_size": [self.dx, self.dy, self.dz], "dimensions": [self.nx, self.ny, self.nz], "p_dimensions": [self.ep_dx, self.ep_dy],
                        "c_patterns": self.n_patterns, "c_planes": self.n_planes, "down_factor": self.dd_factor, 'matrix': self.A}
        torch.save(data_to_save, path)
        self.log_matrix(it, is_original)

    def log_matrix(self, it=100, is_original=False):
        log_path = f"./data/matrices/log/A_original.csv" if is_original else f"./data/matrices/log/A_reduced.csv"
        data_to_save = {
            "it": it, "NA": PhysicalModel.NA, "dx": self.dx, "dy": self.dy, "dz": self.dz, "nx": self.nx, "ny": self.ny, "nz": self.nz,
            "E_dx": PhysicalModel.ep_dx, "E_dy": PhysicalModel.ep_dy, "n_patterns": self.n_patterns, "n_planes": self.n_planes,
            "down_factor": self.dd_factor
        }
        new_df = pd.DataFrame(data_to_save, index=[0])
        new_df.to_csv(log_path, mode='a', header=False, index=False)

    def load_matrix(self, it=0, is_original=True, is_approx = False):
        """ 
        Method: function to load an available matrix together with parameters
        """
        path = f"./data/matrices/original/A_{it}.pt" if is_original else (f"./data/matrices/approximated/A_{it}.pt" if is_approx else f"./data/matrices/reduced/A_{it}.pt")
        loaded_data = torch.load(path)

        PhysicalModel.NA = loaded_data["NA"]
        [self.dx, self.dy, self.dz] = loaded_data["voxel_size"]
        [self.nx, self.ny, self.nz] = loaded_data["dimensions"]
        [self.ep_dx, self.ep_dy] = loaded_data["p_dimensions"]
        self.n_patterns = loaded_data["c_patterns"]
        self.n_planes = loaded_data["c_planes"]
        self.dd_factor = loaded_data["down_factor"]
        self.A = loaded_data["matrix"].to(self.device)

    def visualize_A(self):
        """ 
        Method: function to visualize the matrix in log scale (0 --> 1e-30)
        """
        A_ = self.A.detach().cpu().numpy()
        A_[A_ == 0] = 1e-30
        A_ = np.log(A_)
        show_image(A_, fig_size=(12, 8), title=r"A(Log Scale)")
