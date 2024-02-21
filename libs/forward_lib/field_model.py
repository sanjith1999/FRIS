from libs.forward_lib.physical_model import dmd_patterns, psf_model, conv_3D
import torch
from libs.forward_lib.visualizer import show_planes_z, visualize_SSIM

class FieldModel:
    """ 
     Class: represents the whole forward process of a single photon micrscopy
    """
    
    # Class Variables
    lambda_ = 532.0/1000                            #um
    NA      = .8
    r_index = 1
    dx, dy, dz = 0.08, 0.08, 0.08                   #um
    ep_dx, ep_dy = .64, .64
    w = 4
    
    def __init__(self, nx=4, ny=4, nz=4, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.device = device
        self.init_dmd()

    def __str__(self):
        desc = ""
        desc += "Field Space Specifications\n----------------------------------------------\n"
        desc += f"NA\t\t\t\t: {self.NA}\n"
        desc += f"Space Dimension \t\t: {self.nx*self.dx}um × {self.ny*self.dy}um × {self.nz*self.dz}um\n"
        desc += f"voxel Size \t\t\t: {self.dx}um × {self.dy}um × {self.dz}um\n"
        desc += f"Pattern Dimension \t\t: {self.ep_dx}um × {self.ep_dy}um \n"
        desc += f"Computational Device \t\t: {self.device}"
        return desc


    def init_psf(self):
        """  
        Method: calculate the point spread function and intepret both excitation and emission parts
        """
        LOAD=51
        if LOAD>0:
            psf = (torch.load(f"./data/matrices/field/PSF_{LOAD}.pt")['matrix']).to(self.device)                                        # Manual extra-care should be taken to match parameters
            print("PSF Loaded Successfully...!\n\n")
        else:
            psf = psf_model(self.NA, self.r_index, self.lambda_, self.dx, self.dy, self.dz, self.nx, self.ny, self.nz).to(self.device)
        self.exPSF_3D = psf().detach().permute(0,3,1,2)
        self.emPSF_3D = self.exPSF_3D.abs().square().sum(dim=0).unsqueeze(dim=0).sqrt()
        return 1
    
    def init_dmd(self):
        """ 
        Method: initializing the DMD patterns
        """
        self.dmd = dmd_patterns(self.ep_dx, self.ep_dy, self.dx, self.dy, self.nx, self.ny, self.device)
        self.dmd.initialize_patterns()

    def propagate_field(self):
        """ 
        Method: forward process on an object
        """
        self.init_psf()
        ht_3D = torch.zeros(1, self.nz, self.nx, self.ny).float().to(self.device)               # DMD in 3D
        ht_3D[:, self.nz // 2] = self.dmd.ht_2D_list[0]

        H1 = conv_3D(self.exPSF_3D, ht_3D, self.w)
        self.H2 = H1.abs().square().sum(dim=0).sqrt()                                       # field in the object space
        return 1
    
    def correlation_measure(self, seperation = 1):
        """ 
        Method: calculation of correlation between planes at specified seperation(um)
        """
        corr_list = []
        plane_step  = max(1, round(seperation/self.dz))
        n_planes = int(self.nz//plane_step)
        for p in range(n_planes-1):
            sig1 = self.H2[p*plane_step].flatten()
            sig2 = self.H2[(p+1)*plane_step].flatten()
            sigs = torch.stack((sig1, sig2))
            corr = torch.corrcoef(sigs)[0][1].item()
            corr_list.append(corr)
        visualize_SSIM(measures=[corr_list], x_values=[(p-n_planes//2)*seperation for p in range( n_planes-1)], x_label="Left Plane", y_label="Cross-Correlation", title=f"Plane Seperation: {seperation}um")
        
    

    def save_object_space(self, it = 100):
        """ 
        Method: calculation of correlation between planes at specified seperation(um)
        """
        path = f"./data/matrices/field/H_{it}.pt" 
        data_to_save = {
            "NA"            :   self.NA,
            "voxel_size"    :   [self.dx, self.dy, self.dz],
            "dimensions"    :   [self.nx, self.ny, self.nz],
            "p_dimensions"  :   [self.ep_dx, self.ep_dy],
            "field"         :   self.H2,
            "DMD"           :   self.dmd.ht_2D_list[0]
        }
        torch.save(data_to_save, path)
        log_path = f"./data/matrices/log/H_log.csv"
        log_message = f"{it, self.NA, self.dx, self.dy, self.dz,self.nx, self.ny, self.nz, self.ep_dx, self.ep_dy}"
        with open(log_path, "a") as log_file:
            log_file.write(log_message + "\n")  
    

    def load_object_space(self, it = 0):
        """ 
        Method: calculation of correlation between planes at specified seperation(um)
        """
        path = f"./data/matrices/field/H_{it}.pt" 
        loaded_data = torch.load(path)
        self.NA = loaded_data['NA']
        [self.dx, self.dy, self.dz] = loaded_data['voxel_size']
        [self.nx, self.ny, self.nz] = loaded_data['dimensions']
        [self.ep_dx, self.ep_dy] = loaded_data['p_dimensions']
        self.H2 = loaded_data['field']
        self.init_dmd()
        self.dmd.ht_2D_list[0] = loaded_data['DMD']


    def visualize_at_seperation(self, seperation= 1):
        """ 
        Method: visualizing planes at specified seperation(um)
        """
        plane_step  = max(1, round(seperation/self.dz))
        n_planes = int(self.nz//plane_step)
        show_planes_z(self.H2.detach().cpu().numpy(), title = f"Seperation: {seperation}um", z_planes=[i*plane_step for i in range(n_planes)])