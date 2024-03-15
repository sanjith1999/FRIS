import torch
from torch import nn
from modules.decoders import *
from modules.d2nn_models_new import *

class AE(nn.Module):
    def __init__(self, cfg):
        super(AE, self).__init__()        
        self.d2nn_enc =  d2nn_general(cfg)
        self.dec = eval(cfg['decoder'])(cfg)
    def forward(self, x):
        return self.dec(self.d2nn_enc(x).abs()**2)