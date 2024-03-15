import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.d2nn_models import *
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexBatchNorm1d, ComplexConv2d, ComplexConvTranspose2d, ComplexLinear, ComplexReLU
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d

class complex_cnn(nn.Module):
    def __init__(self,cfg):
        super(complex_cnn,self).__init__()
        
        self.n_i = cfg['img_size']
        self.kernel_size = cfg['kernel_size']
        self.bias_ = cfg['bias']
        
        self.conv1 = ComplexConv2d(in_channels= 1, out_channels= 1, kernel_size=self.kernel_size, stride= 1, padding=self.kernel_size//2, bias=self.bias_)
        self.conv2 = ComplexConv2d(in_channels= 1, out_channels= 1, kernel_size=self.kernel_size, stride= 1, padding=self.kernel_size//2, bias=self.bias_)
        self.conv3 = ComplexConv2d(in_channels= 1, out_channels= 1, kernel_size=self.kernel_size, stride= 1, padding=self.kernel_size//2, bias=self.bias_)
        self.conv4 = ComplexConv2d(in_channels= 1, out_channels= 1, kernel_size=self.kernel_size, stride= 1, padding=self.kernel_size//2, bias=self.bias_)
        self.conv5 = ComplexConv2d(in_channels= 1, out_channels= 1, kernel_size=self.kernel_size, stride= 1, padding=self.kernel_size//2, bias=self.bias_)
        
    def forward(self, input_e_field):
        
        x = input_e_field.view(-1, 1, self.n_i, self.n_i)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        return x[:,0] # Removing the channel dimension
        

class complex_cnn_with_maxpool(nn.Module):
    def __init__(self,cfg):
        super(complex_cnn_with_maxpool,self).__init__()
        
        self.n_i = cfg['img_size']
        self.kernel_size = cfg['kernel_size']
        
        self.conv1 = ComplexConv2d(in_channels= 1, out_channels= 1, kernel_size=self.kernel_size, stride= 1, padding=self.kernel_size//2)
        self.conv2 = ComplexConv2d(in_channels= 1, out_channels= 1, kernel_size=self.kernel_size, stride= 1, padding=self.kernel_size//2)
        self.conv3 = ComplexConv2d(in_channels= 1, out_channels= 1, kernel_size=self.kernel_size, stride= 1, padding=self.kernel_size//2)
        self.conv4 = ComplexConv2d(in_channels= 1, out_channels= 1, kernel_size=self.kernel_size, stride= 1, padding=self.kernel_size//2)
        self.conv5 = ComplexConv2d(in_channels= 1, out_channels= 1, kernel_size=self.kernel_size, stride= 1, padding=self.kernel_size//2)
        
        self.tconv = ComplexConvTranspose2d(in_channels=1, out_channels=1, kernel_size=6, stride=1, padding=0)
        
    def forward(self, input_e_field):
        
        x = input_e_field.view(-1, 1, self.n_i, self.n_i)
        
        x = self.conv1(x) # 64
        x = complex_max_pool2d(x, kernel_size=2, stride=1) # 63
        x = self.conv2(x)
        x = complex_max_pool2d(x, kernel_size=2, stride=1) # 62
        x = self.conv3(x)
        x = complex_max_pool2d(x, kernel_size=2, stride=1) # 61
        x = self.conv4(x)
        x = complex_max_pool2d(x, kernel_size=2, stride=1) # 60
        x = self.conv5(x)
        x = complex_max_pool2d(x, kernel_size=2, stride=1) # 59
        
        x = tconv(x)
        
        return x[:,0] # Removing the channel dimension
        
        

class complex_fc(nn.Module):
    def __init__(self, cfg= None):
        super(complex_fc,self).__init__()
        self.n_i = cfg['img_size']
        
        self.model= nn.Sequential(ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2))
        
    def forward(self, input_e_field):
        x = input_e_field.view(-1, self.n_i**2)
        x = self.model(x).view(-1, self.n_i, self.n_i)
        return x
    
    

class complex_fc_sigmoid(nn.Module):
    def __init__(self, cfg= None):
        super(complex_fc_sigmoid,self).__init__()
        self.n_i = cfg['img_size']
        
        self.model= nn.Sequential(ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2))
        
    def forward(self, input_e_field):
        x = input_e_field.view(-1, self.n_i**2)
        x = self.model(x).view(-1, self.n_i, self.n_i)
        x= F.sigmoid(x.abs())*torch.exp(1j*x.angle())
        return x
    
class complex_fc_nonlinear(nn.Module):
    def __init__(self, cfg= None):
        super(complex_fc_nonlinear,self).__init__()
        self.n_i = cfg['img_size']
        
        self.model= nn.Sequential(ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexReLU(),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexReLU(),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexReLU(),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexReLU(),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2))
        
    def forward(self, input_e_field):
        x = input_e_field.view(-1, self.n_i**2)
        x = self.model(x).view(-1, self.n_i, self.n_i)
        x= F.sigmoid(x.abs())*torch.exp(1j*x.angle())
        return x
    
class complex_fc_nonlinear_nosigmoid(nn.Module):
    def __init__(self, cfg= None):
        super(complex_fc_nonlinear_nosigmoid,self).__init__()
        self.n_i = cfg['img_size']
        
        self.model= nn.Sequential(ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexReLU(),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexReLU(),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexReLU(),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexReLU(),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2))
        
    def forward(self, input_e_field):
        x = input_e_field.view(-1, self.n_i**2)
        x = self.model(x).view(-1, self.n_i, self.n_i)
        return x
    
class complex_fc_nonlinear_batchnorm(nn.Module):
    def __init__(self, cfg= None):
        super(complex_fc_nonlinear_batchnorm,self).__init__()
        self.n_i = cfg['img_size']
        
        self.model= nn.Sequential(ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexBatchNorm1d(num_features= self.n_i**2),
                                  ComplexReLU(),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexBatchNorm1d(num_features= self.n_i**2),
                                  ComplexReLU(),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexBatchNorm1d(num_features= self.n_i**2),
                                  ComplexReLU(),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2),
                                  ComplexBatchNorm1d(num_features= self.n_i**2),
                                  ComplexReLU(),
                                  ComplexLinear(in_features= self.n_i**2, out_features= self.n_i**2))
        
    def forward(self, input_e_field):
        x = input_e_field.view(-1, self.n_i**2)
        x = self.model(x).view(-1, self.n_i, self.n_i)
        x= F.sigmoid(x.abs())*torch.exp(1j*x.angle())
        return x
    
    

class simple_decoder(nn.Module):
    def __init__(self, cfg):
        super(simple_decoder, self).__init__()
        self.img_size= cfg['img_size']
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels= cfg['n_channels'][0], kernel_size= 3, padding=1, stride=1), # 64
            nn.ReLU(),
            nn.Conv2d(in_channels= cfg['n_channels'][0], out_channels= cfg['n_channels'][1], kernel_size= 3, padding=1, stride=1), # 64
            nn.ReLU(),
            nn.Conv2d(in_channels= cfg['n_channels'][1], out_channels= cfg['n_channels'][2], kernel_size= 3, padding=1, stride=1), # 64
        )
  
    def forward(self, x):
        x= x.view(-1, 1, self.img_size, self.img_size)
        return F.sigmoid(self.decoder(x))[:,0] # Removing the channel dimension
    

class AE(nn.Module):
    def __init__(self, cfg, d2nn, decoder):
        super(AE, self).__init__()
        self.d2nn_enc = d2nn(cfg)
        self.dec = decoder(cfg)
    def forward(self, x):
        return self.dec(self.d2nn_enc(x).abs())