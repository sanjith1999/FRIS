import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from torch.nn.functional import conv2d

class BerHu(nn.Module):
    def __init__(self, reduction: str = 'mean', threshold: float = 0.2) -> None :
        '''
            Args:
                reduction (string, optional): Specifies the reduction to apply to the output:
                                              default ('mean')
                threshold (float, optional) : Specifies the threshold at which to change between threshold-scaled L1 and L2 loss.
                                              The value must be positive.  Default: 0.2
                                              (Value based on Mengu et al. https://arxiv.org/abs/2108.07977)
        '''
        
        super(BerHu, self).__init__()
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, X_hat, X):
        '''
            Function to calculate reversed huber distance between predicted and ground truth.
            
            Args:
                X_hat : Predicted image ((n_samples, img_size, img_size), dtype= torch.float)  
                X     : Ground Truth image ((n_samples, img_size, img_size), dtype= torch.float)  
                
            Returns:
                Reversed huber loss (BerHu loss)
        '''
        diff = torch.abs(X-X_hat)

        phi =  torch.std(X, unbiased=False) * self.threshold 

        L1 = -F.threshold(-diff, -phi, 0.)                         # L1 loss for values less than thresh (phi)
        L2 =  F.threshold(diff**2 - phi**2, 0., -phi**2.) + phi**2 # L2 loss for values greater than thresh (phi)


        L2_ = F.threshold(L2, phi**2, -phi**2) + phi**2 # L2 loss + phi^2 for values greater than thresh (phi)
        L2_ = L2_ / (2.*phi) # Equation : (L2 + phi^2)/(2*Phi)

        loss = L1 + L2_

        if(self.reduction ==  'mean'):
            loss = torch.mean(loss)
        else:
            loss = torch.sum(loss)
        return loss
    

class ssim_loss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, k1=0.01, k2=0.03, max_val=1.0):
        
        super(ssim_loss, self).__init__()
        
        self.window_size = window_size
        self.sigma   = sigma
        self.k1      = k1
        self.k2      = k2
        self.max_val = max_val
        
    def create_window(self, window_size, sigma):
        """Create a 2D Gaussian window."""
        window = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        window = window / window.sum()
        return window.unsqueeze(0).unsqueeze(0).repeat((3, 1, 1, 1))
    
    def forward(self, img1, img2):
        # Add channel dimension
        img1 = img1.unsqueeze(1)
        img2 = img2.unsqueeze(1)
        
        # Define weights for the SSIM calculation
        window = self.create_window(self.window_size, self.sigma)
        window = window.to(img1.device)
        mu1 = conv2d(img1, window, padding=self.window_size//2, groups=img1.shape[1])
        mu2 = conv2d(img2, window, padding=self.window_size//2, groups=img2.shape[1])
        mu1_mu2 = mu1 * mu2
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)

        # Calculate the variance and covariance
        sigma1_sq = conv2d(img1 * img1, window, padding=self.window_size//2, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = conv2d(img2 * img2, window, padding=self.window_size//2, groups=img2.shape[1]) - mu2_sq
        sigma12 = conv2d(img1 * img2, window, padding=self.window_size//2, groups=img1.shape[1]) - mu1_mu2

        # Calculate the SSIM
        C1 = (self.k1 * self.max_val) ** 2
        C2 = (self.k2 * self.max_val) ** 2
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        # Return the mean SSIM across all channels
        return 1 - ssim.mean()