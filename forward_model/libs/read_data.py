import torch
import tifffile
from PIL import Image
import numpy as np
import h5py
import random

raw_data_type = "Nothing"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialized Parameters
def init_parameters(Nx_, Ny_, Nz_, dx_, dy_, dz_):
    global X, Nx, Ny, Nz, step_x, step_y, step_z,dx, dy, dz
    dx, dy, dz = dx_, dy_, dz_
    step_x , step_y, step_z = max(1, int(dx_*3)), max(int(dy_*3),1) , max(int(dz_*3),1)
    Nx, Ny, Nz = Nx_, Ny_, Nz_
    X = torch.zeros(1, Nz, Nx, Ny).float().to(device)
    print("Sucessfully Initialized Read Data Parameters...!!!")


# Function to Store
def load_object(object_type="blood_cell",verbose=False, radius = 5):
    try:
        if object_type == "bead":
            load_fluorescense_bead()
        elif object_type == "3D_sphere":
            create_spherical_object(radius = radius)
        elif object_type == "neural_cell":
            load_cell_data(True)
        elif object_type == "blood_cell":
            load_cell_data(False)
        else:
            print("Object type is not available...!!!")
            return -1
        if verbose:
            print("Object Loaded Sucessfully...!!!")
    except:
        print("Failed to Load Object...!!!")
    return 0
    


# Reading the Flourescence Data
def load_fluorescense_bead():
    Iz = 201
    low_l, up_l = round((Iz-Nz)/2), round((Iz+Nz)/2)
    low_l -= 1 if up_l - low_l != Nz else 0

    for slice_index in range(low_l, up_l):
        # Update the path accordingly
        image_path = f'./data/Bead/z{slice_index:04d}.tif'
        image = tifffile.imread(image_path).astype(float)  # Read image as float
        assert image is not None
        r_image = torch.tensor(np.array(Image.fromarray(image).resize((Nx, Ny)))).to(device)

        # Store the image data in the tensor
        X[0, slice_index-low_l, :, :] = r_image


# Creating 3D spehere
def create_spherical_object(radius = 3): # Radius ~ um
    # Create a grid of points in the 3D tensor space
    x = torch.linspace((-Nx)//2+1, Nx//2, Nx)
    y = torch.linspace((-Ny)//2+1, Ny//2, Ny)
    z = torch.linspace((-Nz)//2+1, Nz//2, Nz)

    # Create a meshgrid from the points
    z, x, y = torch.meshgrid(z, x, y)

    # Calculate the distance from each point in the grid to the center
    distance = torch.sqrt((x*dx)**2 + (y*dy)**2 + (z*dz)**2)

    # Create a mask to identify points inside the sphere
    inside_sphere = distance <= radius
    inside_sphere = inside_sphere.unsqueeze(0)
    X[inside_sphere] = 1
    X[~inside_sphere] = 0


# Reading the NeuralCell, BloodCell Data
def load_cell_data(is_neural=True):
    global raw_data_type,raw_data
    if (raw_data_type != "neural_cell" and is_neural):
        with h5py.File('./data/Deep2/PS_SOM_mice_20190317.mat', 'r') as mat_file:
            Data = (mat_file['Data']['cell'])[3,0]
            arr = mat_file[Data][:]
            raw_data = torch.from_numpy(arr).float().to(device)
        raw_data_type = "neural_cell"
    
    elif raw_data_type != "blood_cell" and (not is_neural):
        with h5py.File('./data/Deep2/BV_03102021.mat', 'r') as mat_file:
            Data = (mat_file['Data']['cell'])[:]
            raw_data = torch.from_numpy(Data).to(device)
        raw_data_type = "blood_cell"

    # Picking a Randow Cuboid from Object
    tensor_size = raw_data.shape

    # Generate random coordinates within valid range
    x_start = random.randint(0, tensor_size[1] - Nx*step_x)
    y_start = random.randint(0, tensor_size[2] - Ny*step_y)
    z_start = random.randint(0, tensor_size[0] - Nz*step_z)

    # Extract the cube

    X_ = raw_data[z_start:z_start+Nz*step_z:step_z, x_start:x_start+Nx*step_x:step_x, y_start:y_start+Ny*step_y:step_y]
    X[0,:,:,:] = (X_ - X_.min())/(X_.max()- X_.min()+(1e-10))