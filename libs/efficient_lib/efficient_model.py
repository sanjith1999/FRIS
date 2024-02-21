import torch
from libs.forward_lib.physical_model import psf_model, conv_3D, dmd_patterns


class EfficientModel:
    """ 
     Class: Efficient Forward Process
    """
    
    # Class Variables
    lambda_ = 532.0/1000                            #um
    NA      = .8
    r_index = 1
    dx, dy, dz = 0.08, 0.08, 0.08                   #um
    ep_dx, ep_dy = .64, .64
    
    def __init__(self, nx, ny, nz, n_patterns , dd_factor = 1, n_planes = 1,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.n_patterns = n_patterns
        self.device = device
        self.dd_factor = dd_factor
        self.n_planes = n_planes
        self.m_planes = [(i*nz)//(n_planes+1) for i in range(1,n_planes+1)]
        self.w = 2
        self.init_psf()
        self.init_dmd()

    def __str__(self):
        desc = ""
        desc += "Forward Model Specifications\n----------------------------------------------\n"
        desc += f"Space Dimension \t\t: {self.nx*self.dx}um × {self.ny*self.dy}um × {self.nz*self.dz}um\n"
        desc += f"voxel Size \t\t\t: {self.dx}um × {self.dy}um × {self.dz}um\n"
        desc += f"DMD Patterns \t\t\t: {self.n_patterns}\n"
        desc += f"Measurement Plane\t\t: {self.m_planes}\n"
        desc += f"Detector Pool size \t\t: {self.dd_factor}×{self.dd_factor}\n"
        desc += f"Computational Device \t\t: {self.device}"
        return desc


    def init_psf(self):
        """  
        Method: calculate the point spread function and intepret both excitation and emission parts
        """
        LOAD=-1
        if LOAD>0:
            psf = (torch.load(f"./data/matrices/field/PSF_{LOAD}.pt")['matrix']).to(self.device)                                        # Manual extra-care should be taken to match parameters
            print("PSF Loaded Successfully...!\n\n")
        else:
            psf = psf_model(self.NA, self.r_index, self.lambda_, self.dx, self.dy, self.dz, self.nx, self.ny, self.nz).to(self.device)
        self.exPSF_3D = psf().detach().permute(0,3,1,2)
        self.emPSF_3D = self.exPSF_3D.abs().square().sum(dim=0).sqrt().unsqueeze(dim=0)
        return 1
    
    def init_dmd(self):
        """ 
        Method: initializing the DMD patterns
        """
        self.dmd = dmd_patterns(self.ep_dx, self.ep_dy, self.dx, self.dy, self.nx, self.ny, self.device)
        self.dmd.initialize_patterns(self.n_patterns)

    def propagate_dmd(self, p_no = 1):
        """ 
        Method: forwarding DMD pattern
        """
        ht_3D = torch.zeros(1, self.nz, self.nx, self.ny).float().to(self.device)               # DMD in 3D
        if p_no > self.dmd.n_patterns:
            print("Not enough DMD patterns...!")
        ht_3D[:, self.nz // 2] = self.dmd.ht_2D_list[p_no-1]

        H1 = conv_3D(self.exPSF_3D, ht_3D, w=self.w)
        self.H2 = H1.abs().square().sum(dim=0).sqrt()                                                                       

    def propagate_object(self, cor = (0,0,0)):
            ix, iy, iz = cor
            cx, cy = self.nx//2, self.ny//2
            det_Y = torch.zeros(self.nx, self.ny).to(self.device).float()

            i_iz = self.nz-iz
            l_ix, l_iy = min(cx, ix), min(cy, iy)
            r_ix, r_iy = min(self.nx-ix, cx), min(self.ny-iy, cy) 

            if i_iz < self.nz:
                det_Y[ix-l_ix:ix+r_ix, iy-l_iy: iy+r_iy] = self.H2[iz, ix, iy]* self.emPSF_3D[0, i_iz,cx-l_ix: cx+r_ix, cy-l_iy:cy+r_iy ]
            scale_factor = (1/self.dd_factor, 1/self.dd_factor)
            det_Y = torch.nn.functional.interpolate(det_Y.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor, mode='area').squeeze()
            return det_Y
    

