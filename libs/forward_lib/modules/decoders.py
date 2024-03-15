import torch
from torch import nn


class simple_decoder(nn.Module):
    def __init__(self, cfg):
        super(simple_decoder, self).__init__()
        self.img_size= cfg['img_size']
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels= 3, kernel_size= 3, padding=1, stride=1), # 64
            nn.ReLU(),
            nn.Conv2d(in_channels= 3, out_channels= 3, kernel_size= 3, padding=1, stride=1), # 64
            nn.ReLU(),
            nn.Conv2d(in_channels= 3, out_channels= 1, kernel_size= 3, padding=1, stride=1), # 64
            nn.Sigmoid())
  
    def forward(self, x):
        x= x.view(-1, 1, self.img_size, self.img_size)
        return self.decoder(x)[:,0] # Removing the channel dimension
    