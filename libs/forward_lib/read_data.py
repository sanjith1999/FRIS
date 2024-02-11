import torch
import torch.nn.functional as F
import tifffile
from PIL import Image
import numpy as np
import h5py
import random
from libs.forward_lib.linearized_process import LinearizedModel
from libs.forward_lib.physical_model import PhysicalModel
from libs.forward_lib.visualizer import show_planes_z

class ReadData:
    """ 
    Class: handle the reading data functionality from stored data values
    """

    # class varibles
    device = LinearizedModel.device
    dx, dy, dz = PhysicalModel.dx, PhysicalModel.dy, PhysicalModel.dz
    v_nx, v_ny, v_nz = LinearizedModel.v_nx, LinearizedModel.v_ny, LinearizedModel.v_nz

    def __init__(self, nx, ny, nz):
        self.raw_data_type = "Nothing"   
        self.nx, self.ny, self.nz = nx, ny, nz
        self.X = torch.zeros(1, nx, ny, nz).to(self.device)

    def __str__(self):
        desc  = ""
        desc += f"Spatial Dimension\t: {self.nx}×{self.ny}×{self.nz}\n"
        desc += f"Reduced Dimension\t: {self.nx//self.v_nx}×{self.ny//self.v_ny}×{self.nz//self.v_nz}\n"
        desc += f"Raw Data Type\t\t: {self.raw_data_type}\n"
        desc += f"Device\t\t\t: {self.device}\n"
        return desc

    def load_object(self, object_type="blood_cell",verbose=False, radius = 5):
        """ 
        Method: load the specified object
        """
        try:
            if object_type == "bead":
                self.load_fluorescense_bead()
            elif object_type == "3D_sphere":
                self.create_spherical_object(radius = radius)
            elif object_type == "neural_cell":
                self.load_cell_data(True)
            elif object_type == "blood_cell":
                self.load_cell_data(False)
            else:
                print("Object type is not available...!!!")
            if verbose:
                print("Object Loaded Sucessfully...!!!")
        except:
            print("Failed to Load Object...!!!")
    


    def load_fluorescense_bead(self):
        """
        Method: assitant to load the fluorescense bead object 
        """
        Iz = 201
        low_l, up_l = round((Iz-self.nz)/2), round((Iz+self.nz)/2)
        low_l -= 1 if up_l - low_l != self.nz else 0

        for slice_index in range(low_l, up_l):
            # Update the path accordingly
            image_path = f'./data/Bead/z{slice_index:04d}.tif'
            image = tifffile.imread(image_path).astype(float)  # Read image as float
            assert image is not None
            r_image = torch.tensor(np.array(Image.fromarray(image).resize((self.nx, self.nz)))).to(device)

            # Store the image data in the tensor
            self.X[0, slice_index-low_l, :, :] = r_image


    def create_spherical_object(self, radius = 3): # Radius ~ um
        """ 
        Method: assistant to create spherical object
        """
        # Create a grid of points in the 3D tensor space
        x = torch.linspace((-self.nx)//2+1, self.nx//2, self.nx)
        y = torch.linspace((-self.ny)//2+1, self.ny//2, self.ny)
        z = torch.linspace((-self.nz)//2+1, self.nz//2, self.nz)

        # Create a meshgrid from the points
        z, x, y = torch.meshgrid(z, x, y, indexing = 'ij')

        # Calculate the distance from each point in the grid to the center
        distance = torch.sqrt((x*self.dx)**2 + (y*self.dy)**2 + (z*self.dz)**2)

        # Create a mask to identify points inside the sphere
        inside_sphere = distance <= radius
        inside_sphere = inside_sphere.unsqueeze(0)
        self.X[inside_sphere] = 1
        self.X[~inside_sphere] = 0


    def load_cell_data(self, is_neural=True):
        """ 
        Method: assistant to load either blood cell or neural cell data
        """
        if (self.raw_data_type != "neural_cell" and is_neural):
            with h5py.File('./data/Deep2/PS_SOM_mice_20190317.mat', 'r') as mat_file:
                Data = (mat_file['Data']['cell'])[3,0]
                arr = mat_file[Data][:]
                self.raw_data = torch.from_numpy(arr).float().to(self.device)
            self.raw_data_type = "neural_cell"
        
        elif self.raw_data_type != "blood_cell" and (not is_neural):
            with h5py.File('./data/Deep2/BV_03102021.mat', 'r') as mat_file:
                Data = (mat_file['Data']['cell'])[:]
                self.raw_data = torch.from_numpy(Data).to(self.device)
            self.raw_data_type = "blood_cell"

        # Picking a Randow Cuboid from Object
        tensor_size = self.raw_data.shape

        # Generate random coordinates within valid range
        x_start = random.randint(0, tensor_size[1] - self.nx)
        y_start = random.randint(0, tensor_size[2] - self.ny)
        z_start = random.randint(0, tensor_size[0] - self.nz)

        # Extract the cube

        X_ = self.raw_data[z_start:z_start+self.nz, x_start:x_start+self.nx, y_start:y_start+self.ny]
        self.X[0,:,:,:] = (X_ - X_.min())/(X_.max()- X_.min()+(1e-10))

    def reduce_dimension(self):
        """ 
        Method: reduce the dimension of X through 3D average pooling
        """
        kernel_size = (self.v_nz, self.v_nx, self.v_ny)  
        self.X_r = F.avg_pool3d(self.X, kernel_size)
    

    def visualize_data(self, is_orginal = False, n_planes = 8):
        """ 
        Method: visualize X or X_r
        """
        if is_orginal:
            X_ = self.X[0].detach().cpu()
            title = "High-Dimensional Object"
            planes = [itz*(self.nz//n_planes) for itz in range(n_planes)]
        else:
            X_ = self.X_r[0].detach().cpu()
            title = "Pooled Object"
            r_nz = self.nz//self.v_nz
            planes = [itz*(r_nz//n_planes) for itz in range(n_planes)]

        show_planes_z(X_,title=title,z_planes=planes )
    
    def store_data(self, fol_path = "./data/dataset/", it = 0, is_orginal = False):
        """ 
        Method: store the reduced dimensional data at specified location
        """
        if is_orginal:
            torch.save(self.X, fol_path+f"X_{it}.pt")
        else:
            torch.save(self.X_r, fol_path+f"X_{it}.pt")