import torchvision
import torch
from random import randint
import libs.forward_lib.visualizer as vs
from libs.forward_lib.linearized_process import LinearizedModel
# import torch.nn.functional as F
# padding = (mz//2, mz//2, mx//2, mx//2, my//2, my//2)
# F.pad(tensor, padding, "constant", value=0)



class MnistSimulator:
    device = LinearizedModel.device
    int_weight = .3
    mx, my, mz = 28, 28, 28

    def __init__(self, nx, ny, nz, n_bodies = 1):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.n_bodies = n_bodies
        self.X = torch.zeros(nz, nx, ny).to(self.device)
        self.load_mnist_data()

    def __str__(self):
        desc  = ""
        desc += f"MNIST Simulator\n"
        desc += f"--------------------------------------------\n"
        desc += f"Spatial Dimension\t\t: {self.nx}×{self.ny}×{self.nz}\n"
        desc += f"Number of Bodies \t\t: {self.n_bodies}\n"
        desc += f"Original Intensity Weight \t: {self.int_weight}\n"
        return desc


    def update_data(self):
        self.X[:, :, :] = 0
        for i in range(self.n_bodies):
            sx, sy, sz = randint(0, self.nx-self.mx), randint(0, self.ny-self.my), randint(0, self.nz-self.mz)
            self.X[sz:sz+self.mz, sx:sx+self.mx, sy:sy+self.my] = self.augmented_mnist_body()
        

    def load_mnist_data(self):
        self.mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
        self.mnist_limit = len(self.mnist_trainset)
        self.transform = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])

    def augmented_mnist_body(self):
        body_id = randint(0, self.mnist_limit)
        angle_x, angle_y, angle_z = randint(-90, 90), randint(-90, 90), randint(-90, 90)

        body = self.transform(self.mnist_trainset[body_id][0]).to(self.device)
        s_body = body.repeat(28, 1, 1)
        s_body[0:5, :, :] , s_body[23:28, :, :] = 0, 0
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
    
    def visualize_object(self, ele_ang=10, azim_ang = 40):
        vs.vis_3d(self.X.detach().cpu(), elev_ang=ele_ang, azim_ang=azim_ang)
        
    