import torchvision
import torch
from random import randint
import numpy as np
from scipy.spatial.transform import Rotation
import libs.forward_lib.visualizer as vs
from libs.forward_lib.linearized_process import LinearizedModel
from libs.forward_lib.physical_model import PhysicalModel



class MnistSimulator:
    device = LinearizedModel.device
    int_weight = .3
    mx, my, mz = 28, 28, 28

    def __init__(self, nx, ny, nz, n_bodies = 1, up_factor = 1):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.n_bodies = n_bodies
        self.up_factor = up_factor
        self.X = torch.zeros(nz, nx, ny).to(self.device)
        self.uSampler = torch.nn.Upsample(scale_factor=up_factor, mode='trilinear')
        self.load_mnist_data()

    def __str__(self):
        desc  = ""
        desc += f"MNIST Simulator\n"
        desc += f"--------------------------------------------\n"
        desc += f"Spatial Dimension\t\t: {self.nx}×{self.ny}×{self.nz}\n"
        desc += f"Number of Bodies \t\t: {self.n_bodies}\n"
        desc += f"Original Intensity Weight \t: {self.int_weight}\n"
        desc += f"Upsampling Factor\t\t: {self.up_factor}\n"
        return desc


    def update_data(self):
        self.X[:, :, :] = 0
        for i in range(self.n_bodies):
            ux, uy , uz  = self.mx*self.up_factor, self.my*self.up_factor, self.mz*self.up_factor
            sx, sy, sz = randint(0, self.nx-ux), randint(0, self.ny-uy), randint(0, self.nz-uz)
            self.X[sz:sz+uz, sx:sx+ux, sy:sy+uy] = self.uSampler(self.augmented_mnist_body().unsqueeze(0).unsqueeze(0)).squeeze()
        

    def load_mnist_data(self):
        self.mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
        self.mnist_limit = len(self.mnist_trainset)
        self.transform = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])

    def augmented_mnist_body(self):
        body_id = randint(0, self.mnist_limit)
        angle_x, angle_y, angle_z = randint(-90, 90), randint(-90, 90), randint(-90, 90)

        body = self.transform(self.mnist_trainset[body_id][0]).to(self.device)
        s_body = body.repeat(28, 1, 1)
        s_body[0:4, :, :] , s_body[24:28, :, :] = 0, 0
        r_body = self.rotate_3d_image(s_body, angle_x, angle_y, angle_z)
        smooth_body = self.normalize((r_body**self.int_weight)*self.intensity_smoother())
        return smooth_body
    
    def intensity_smoother(self):
        cx, cy, cz = randint(0, self.mx), randint(0, self.my), randint(0, self.mz)
        x = torch.linspace(0, self.mx, self.mx)
        y = torch.linspace(0, self.my, self.my)
        z = torch.linspace(0, self.mz, self.mz)
        z, x, y = torch.meshgrid(z, x, y, indexing = 'ij')
        distance = torch.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
        return distance.to(self.device)


    def rotate_3d_image(self, image_3d,angle_x,angle_y,angle_z):
        rot_img = torchvision.transforms.functional.rotate(image_3d,angle = angle_z)
        rot_img = torchvision.transforms.functional.rotate(rot_img.permute(1,0,2),angle = angle_x)
        rot_img = torchvision.transforms.functional.rotate(rot_img.permute(2,1,0),angle = angle_y)
        return rot_img.permute(1, 2, 0)
    
    def normalize(self, object):
        return (object - object.min())/(object.max()-object.min())
    
    def visualize_object(self, ele_ang=10, azim_ang = 40, vis_planes = False):
        if vis_planes:
            vs.show_planes(self.X.detach().cpu(),title="Object" ,N_z=self.nz)
        else:
            vs.vis_3d(self.X.detach().cpu(), elev_ang=ele_ang, azim_ang=azim_ang)
        

class SyntheticBeadSimulator:
    device = LinearizedModel.device
    dx, dy, dz = PhysicalModel.dx, PhysicalModel.dy, PhysicalModel.dz

    def __init__(self, nx, ny, nz):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.X = torch.zeros(nz, ny, nx).to(self.device)
        self.n_spheres = np.random.randint(1, 4)
        self.n_ellipsoids = np.random.randint(1, 4)
        self.generate_volume()

    def __str__(self):
        desc  = ""
        desc += f"Synthetic bead Simulator\n"
        desc += f"--------------------------------------------\n"
        desc += f"Spatial Dimension\t\t: {self.nx}×{self.ny}×{self.nz}\n"
        desc += f"Number of spheres \t\t: {self.n_spheres}\n"
        desc += f"Number of ellipsoids \t\t: {self.n_ellipsoids}\n"
        return desc

    def generate_rotation_matrix(self, alpha, beta, gamma):
        R_x = Rotation.from_euler('x', alpha)
        R_y = Rotation.from_euler('y', beta)
        R_z = Rotation.from_euler('z', gamma)
        return R_z.as_matrix() @ R_y.as_matrix() @ R_x.as_matrix()
        
    def generate_volume_with_sphere(self, r, center_x, center_y, center_z):
        Z, Y, X = np.mgrid[0:self.nz, 0:self.ny, 0:self.nx]
        distances = np.sqrt(((X - center_x)*self.dx)**2 +
                            ((Y - center_y)*self.dy)**2 +
                            ((Z - center_z)*self.dz)**2)
        normalized_distances = 1 - distances/r
        volume = np.where(distances <= r, normalized_distances, 0)
        return volume
        
    def generate_volume_with_ellipsoid(self, rx, ry, rz, center_x, center_y, center_z):
        volume = np.zeros((self.nz, self.ny, self.nx))
        alpha = np.random.uniform(0, np.pi/4) #around z
        beta = np.random.uniform(0, np.pi/4) #around y
        gamma = np.random.uniform(0, np.pi/4) #around x
        rotation_matrix = self.generate_rotation_matrix(alpha, beta, gamma)
        for z in range(self.nz):
            for y in range(self.ny):
                for x in range(self.nx):
                    point = np.array([z, y, x]) - np.array([center_z, center_y, center_x])
                    rotated_point = (rotation_matrix @ point) * (np.array([self.dz, self.dy, self.dx])) / (np.array([rz, ry, rx]))
                    distance = np.sqrt(np.sum(rotated_point**2))
                    if distance <= 1:
                        volume[z, y, x] = 1 - distance
        return volume
    
    def generate_volume(self):
        volume = np.zeros((self.nz, self.ny, self.nx))
        r_range = (self.nx*self.dx/9, self.nx*self.dx/3)
        for _ in range(self.n_spheres):
            center_x = np.random.randint(0, self.nx)
            center_y = np.random.randint(0, self.ny)
            center_z = np.random.randint(0, self.nz)
            r = np.random.uniform(r_range[0], r_range[1])
            sphere_volume = self.generate_volume_with_sphere(r, center_x, center_y, center_z)
            volume = np.maximum(volume, sphere_volume)
        for _ in range(self.n_ellipsoids):
            center_x = np.random.randint(0, self.nx)
            center_y = np.random.randint(0, self.ny)
            center_z = np.random.randint(0, self.nz)
            rx = np.random.uniform(r_range[0], r_range[1])
            ry = np.random.uniform(r_range[0], r_range[1])
            rz = np.random.uniform(r_range[0], r_range[1])
            sphere_volume = self.generate_volume_with_ellipsoid(rx, ry, rz, center_x, center_y, center_z)
            volume = np.maximum(volume, sphere_volume)
        self.X = torch.from_numpy(volume)

    def visualize_object3D(self, ele_ang=10, azim_ang = 40):
        vs.vis_3d(self.X.detach().cpu(), elev_ang=ele_ang, azim_ang=azim_ang)

    def visualize_object2D(self, n_planes=8):
        z_planes = [itz*(self.nz//n_planes) for itz in range(n_planes)]
        vs.show_planes_z(self.X.detach().cpu(), "plane-by-plane", z_planes=z_planes)