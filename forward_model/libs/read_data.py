import torch
import tifffile
from PIL import Image
import numpy as np
import h5py
import random

raw_data_type = "Nothing"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialized Parameters
def init_parameters(Nx_, Ny_, Nz_):
    global X, Nx, Ny, Nz
    Nx, Ny, Nz = Nx_, Ny_, Nz_
    X = torch.zeros(1, Nz, Nx, Ny).float().to(device)
    print("Sucessfully Initialized Read Data Parameters...!!!")


# Function to Store
def load_object(object_type="blood_cell"):
    try:
        if object_type == "bead":
            load_fluorescense_bead()
        elif object_type == "3D_sphere":
            create_spherical_object()
        elif object_type == "neural_cell":
            load_cell_data(True)
        elif object_type == "blood_cell":
            load_cell_data(False)
        else:
            print("Object type is not available...!!!")
            return -1
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
        r_image = torch.tensor(
            np.array(Image.fromarray(image).resize((Nx, Ny)))).to(device)

        # Store the image data in the tensor
        X[0, slice_index-low_l, :, :] = r_image


# Creating 3D spehere
def create_spherical_object():
    # Create a grid of points in the 3D tensor space
    x = torch.linspace(-1, 1, Nx)
    y = torch.linspace(-1, 1, Ny)
    z = torch.linspace(-1, 1, Nz)

    # Create a meshgrid from the points
    x, y, z = torch.meshgrid(z, x, y)

    # Define the center and radius of the sphere
    center = torch.tensor([0.0, 0.0, 0.0])
    radius = 0.5

    # Calculate the distance from each point in the grid to the center
    distance = torch.sqrt(
        (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

    # Create a mask to identify points inside the sphere
    inside_sphere = distance <= radius
    inside_sphere = inside_sphere.unsqueeze(0)
    X[inside_sphere] = 1


# Reading the NeuralCell, BloodCell Data
def load_cell_data(is_neural=True):
    global raw_data_type
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
    print(Nx,Ny,Nz)

    # Generate random coordinates within valid range
    x_start = random.randint(0, tensor_size[1] - Nx)
    y_start = random.randint(0, tensor_size[2] - Ny)
    z_start = random.randint(0, tensor_size[0] - Nz)

    # Extract the cube
    X[0,:,:,:] = raw_data[z_start:z_start+Nz, x_start:x_start+Nx, y_start:y_start+Ny]
    X = (X - X.min())/(X.max()- X.min()+(1e-12))