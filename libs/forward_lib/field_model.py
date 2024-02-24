from libs.forward_lib.physical_model import PhysicalModel
import torch
from libs.forward_lib.visualizer import show_planes_z, visualize_SSIM


# noinspection PyAttributeOutsideInit
class FieldModel:
    """ 
     Class: represents the whole forward process of a single photon microscopy
    """

    def __init__(self, nx=4, ny=4, nz=4, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.device = device
        self.PM = PhysicalModel(self.nx, self.ny, self.nz, n_patterns=1)

    def __str__(self):
        desc = ""
        desc += "Field Space Specifications\n----------------------------------------------\n"
        desc += f"NA\t\t\t\t: {self.PM.NA}\n"
        desc += f"Space Dimension \t\t: {self.nx * self.PM.dx}um × {self.ny * self.PM.dy}um × {self.nz * self.PM.dz}um\n"
        desc += f"voxel Size \t\t\t: {self.PM.dx}um × {self.PM.dy}um × {self.PM.dz}um\n"
        desc += f"Pattern Dimension \t\t: {self.PM.ep_dx}um × {self.PM.ep_dy}um \n"
        desc += f"Computational Device \t\t: {self.device}"
        return desc

    def propagate_field(self):
        self.PM.init_models()
        self.PM.propagate_dmd()
        self.H2 = self.PM.H2.to(self.device)

    def correlation_measure(self, separation=1):
        """ 
        Method: calculation of correlation between planes at specified separation(um)
        """
        corr_list = []
        plane_step = max(1, round(separation / self.PM.dz))
        n_planes = int(self.nz // plane_step)
        for p in range(n_planes - 1):
            sig1 = self.H2[p * plane_step].flatten()
            sig2 = self.H2[(p + 1) * plane_step].flatten()
            sigs = torch.stack((sig1, sig2))
            corr = torch.corrcoef(sigs)[0][1].item()
            corr_list.append(corr)
        visualize_SSIM(measures=[corr_list], x_values=[(p - n_planes // 2) * separation for p in range(n_planes - 1)], x_label="Left Plane", y_label="Cross-Correlation",
                       title=f"Plane Separation: {separation}um")

    def save_object_space(self, it=100):
        """ 
        Method: calculation of correlation between planes at specified separation(um)
        """
        path = f"./data/matrices/field/H_{it}.pt"
        data_to_save = {
            "NA": self.PM.NA,
            "voxel_size": [self.PM.dx, self.PM.dy, self.PM.dz],
            "dimensions": [self.nx, self.ny, self.nz],
            "p_dimensions": [self.PM.ep_dx, self.PM.ep_dy],
            "field": self.H2,
            "DMD": self.PM.dmd.ht_2D_list[0]
        }
        torch.save(data_to_save, path)
        log_path = f"./data/matrices/log/H_log.csv"
        log_message = f"{it, self.PM.NA, self.PM.dx, self.PM.dy, self.PM.dz, self.nx, self.ny, self.nz, self.PM.ep_dx, self.PM.ep_dy}"
        with open(log_path, "a") as log_file:
            log_file.write(log_message + "\n")

    def load_object_space(self, it=0):
        """ 
        Method: calculation of correlation between planes at specified separation(um)
        """
        path = f"./data/matrices/field/H_{it}.pt"
        loaded_data = torch.load(path)
        self.PM.NA = loaded_data['NA']
        [self.PM.dx, self.PM.dy, self.PM.dz] = loaded_data['voxel_size']
        [self.nx, self.ny, self.nz] = loaded_data['dimensions']
        [self.PM.ep_dx, self.PM.ep_dy] = loaded_data['p_dimensions']
        self.H2 = loaded_data['field']
        self.PM.dmd.ht_2D_list[0] = loaded_data['DMD']

    def visualize_at_separation(self, separation=1):
        """ 
        Method: visualizing planes at specified separation(um)
        """
        plane_step = max(1, round(separation / self.PM.dz))
        n_planes = int(self.nz // plane_step)
        show_planes_z(self.H2.detach().cpu().numpy(), title=f"Separation: {separation}um", z_planes=[i * plane_step for i in range(n_planes)])
