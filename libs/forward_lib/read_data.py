import torch
import torch.nn.functional as F
import tifffile
from PIL import Image
import numpy as np
import h5py
import random
from scipy.spatial.transform import Rotation
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
        self.X = torch.zeros(1, nz, ny, nx).to(self.device)

    def __str__(self):
        desc  = "Read Object Parameters\n"
        desc += "--------------------------\n"
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
            elif object_type == "synthetic_bead":
                self.create_synthetic_bead()
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

    def create_synthetic_bead(self): 
        """ 
        Method: assistant to create synthetic bead object
        """
        def generate_rotation_matrix(alpha, beta, gamma):
            R_x = Rotation.from_euler('x', alpha)
            R_y = Rotation.from_euler('y', beta)
            R_z = Rotation.from_euler('z', gamma)
            return R_z.as_matrix() @ R_y.as_matrix() @ R_x.as_matrix()
        
        def generate_volume_with_sphere(nx, ny, nz, dx, dy, dz, r, center_x, center_y, center_z):
            Z, Y, X = np.mgrid[0:nz, 0:ny, 0:nx]
            distances = np.sqrt(((X - center_x)*dx)**2 +
                                ((Y - center_y)*dy)**2 +
                                ((Z - center_z)*dz)**2)
            normalized_distances = 1 - distances/r
            volume = np.where(distances <= r, normalized_distances, 0)
            return volume
        
        def generate_volume_with_ellipsoid(nx, ny, nz, dx, dy, dz, rx, ry, rz, center_x, center_y, center_z):
            volume = np.zeros((nz, ny, nx))
            alpha = np.random.uniform(0, np.pi/4) #around z
            beta = np.random.uniform(0, np.pi/4) #around y
            gamma = np.random.uniform(0, np.pi/4) #around x
            rotation_matrix = generate_rotation_matrix(alpha, beta, gamma)
            for z in range(nz):
                for y in range(ny):
                    for x in range(nx):
                        point = np.array([z, y, x]) - np.array([center_z, center_y, center_x])
                        rotated_point = (rotation_matrix @ point) * (np.array([dz, dy, dx])) / (np.array([rz, ry, rx]))
                        distance = np.sqrt(np.sum(rotated_point**2))
                        if distance <= 1:
                            volume[z, y, x] = 1 - distance
            return volume
        
        volume = np.zeros((self.nz, self.ny, self.nx))
        num_spheres = np.random.randint(2, 4)
        num_ellipsoids = np.random.randint(2, 4)
        r_range = (self.nx*self.dx/9, self.nx*self.dx/3)
        for _ in range(num_spheres):
            center_x = np.random.randint(0, self.nx)
            center_y = np.random.randint(0, self.ny)
            center_z = np.random.randint(0, self.nz)
            r = np.random.uniform(r_range[0], r_range[1])
            sphere_volume = generate_volume_with_sphere(self.nx, self.ny, self.nz, 
                                                        self.dx, self.dy, self.dz, 
                                                        r, center_x, center_y, center_z)
            volume = np.maximum(volume, sphere_volume)
        for _ in range(num_ellipsoids):
            center_x = np.random.randint(0, self.nx)
            center_y = np.random.randint(0, self.ny)
            center_z = np.random.randint(0, self.nz)
            rx = np.random.uniform(r_range[0], r_range[1])
            ry = np.random.uniform(r_range[0], r_range[1])
            rz = np.random.uniform(r_range[0], r_range[1])
            sphere_volume = generate_volume_with_ellipsoid(self.nx, self.ny, self.nz, 
                                                           self.dx, self.dy, self.dz, 
                                                           rx, ry, rz, center_x, center_y, center_z)
            volume = np.maximum(volume, sphere_volume)
        
        self.X = torch.from_numpy(volume).unsqueeze(0)


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