import torch
from tqdm import tqdm
import numpy as np

class DetectorModel:
    """ 
    Class: detector model that comes just after the forward process
    """
    max_photons     = 10000
    N_reps          = 100
    det_Ymin        = 0
    det_Ymax        = 5000

    def __init__(self, cam_dydt_dark = .005, cam_t_exp = 1, cam_alpha = .01, sigma_read = .005, EM_gain_stages = 32, device = 'cpu'):
        self.cam_dydt_dark  = cam_dydt_dark
        self.cam_t_exp      = cam_t_exp
        self.cam_alpha      = cam_alpha
        self.cam_sigma_read = sigma_read
        self.cam_EM_gain    = EM_gain_stages
        self.device         = device

    def __str__(self):
        desc  = ""
        desc += "Detector Specification\n"
        desc += "--------------------------\n"
        desc += f"Photon Limit\t\t: {self.max_photons}\n"
        desc += f"Dark Current\t\t: {self.cam_dydt_dark}\n"
        desc += f"Exposure Time\t\t: {self.cam_t_exp}\n"
        desc += f"Alpha\t\t\t: {self.cam_alpha}\n"
        desc += f"Read Noise Variance\t: {self.cam_sigma_read}\n"
        desc += f"Number of EM Stages\t: {self.cam_EM_gain}\n"
        return desc



    def simulate_EM_process(self, it =0):
        in_photons = torch.arange(self.max_photons*2)                       # Ask Farhath
        self.em_hist = in_photons.repeat(self.N_reps, 1).t()

        for i in tqdm(range(1, self.cam_EM_gain), desc = "Completed EM Stages: "):
            self.em_hist = self.em_hist + torch.binomial(self.em_hist.type(torch.float64), torch.tensor(self.cam_alpha).type(torch.float64))

        data_to_save = {
            "cam_alpha"     : self.cam_alpha,
            "cam_EM_gain"   : self.cam_EM_gain,
            "max_photons"   : self.max_photons,
            "N_reps"        : self.N_reps,
            "EM_hist"       : self.em_hist
        }

        torch.save(data_to_save, f"./data/matrices/noise/em_hist_{it}.pt")
        print("Successfully Simulated EM Stages...!")

    def load_EM_process(self,it=0):
        loaded_data = torch.load(f"./data/matrices/noise/em_hist_{it}.pt")
        self.cam_alpha      = loaded_data["cam_alpha"]
        self.cam_EM_gain    = loaded_data["cam_EM_gain"]
        self.max_photons    = loaded_data["max_photons"]
        self.N_reps         = loaded_data["N_reps"]
        self.em_hist        = loaded_data["EM_hist"].to(self.device)

    def add_noise(self, det_Y, direct_model = False):
        det_Y = (det_Y - self.det_Ymin)*self.max_photons/(self.det_Ymax - self.det_Ymin)
        det_Y = det_Y.type(torch.long)
        if direct_model:
            return self.direct_noise_model(det_Y)
        return self.noise_model(det_Y)
    
    def noise_model(self, det_Y):

        # Dark Noise Model
        Ydark   = self.cam_dydt_dark * self.cam_t_exp
        noise_Y    = torch.poisson(det_Y + Ydark)        

        for i in tqdm(range(noise_Y.shape[0]), desc = "Completed Calculations: "):
            noise_Y = noise_Y.type(torch.long)
            noise_Y[i] = np.random.choice(self.em_hist[noise_Y[i]].detach().cpu().numpy())
        
        noise_Y[noise_Y > self.max_photons] = self.max_photons
                
        noise_Y    = (noise_Y + torch.normal(mean = 0.,std=self.cam_sigma_read,size = noise_Y.shape).to(self.device))

        return noise_Y



    def direct_noise_model(self, det_Y):
        # Dark Noise Model
        Ydark   = self.cam_dydt_dark * self.cam_t_exp
        noise_Y    = torch.poisson(det_Y + Ydark)
        noise_Y[noise_Y > self.max_photons] = self.max_photons

        for i in tqdm(range(1, self.cam_EM_gain), desc = "Completed EM Stages: "):
            EM_gain = torch.binomial(noise_Y , torch.tensor([self.cam_alpha]).to(self.device))
            noise_Y = noise_Y +  EM_gain

        noise_Y    = (noise_Y + torch.normal(mean = 0., std=self.cam_sigma_read,size = noise_Y.shape).to(self.device))

        return noise_Y