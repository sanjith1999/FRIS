import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go

class mlsq_func(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, k, n, l, u, alpha, train):
        device = x.device
        out = torch.zeros(x.shape).to(device)
        
        for j in range(len(n)):
            if j == 0:
                l1 = l
            else:
                l1 = u[j-1].to(device)
                
            u1 = u[j].to(device)
            n1 = n[j]

            if n == 1:
                step = 0
            else:
                step = (u1 - l1)/(n1 - 1)
            quant_width = step
            start_x = l1 + step/2

            i = torch.arange(n1-1).reshape(n1-1,1).unsqueeze(2).to(device)
            if train:
                out = out + torch.sum(step*torch.sigmoid(k*(x - start_x - i*quant_width)),dim=0)
            else:
                out = out + torch.sum(step*torch.round(torch.sigmoid(k*(x - start_x - i*quant_width))),dim=0)
                
        out = out + l
                
        ctx.save_for_backward(x)
        ctx.u = u
        ctx.l = l
        ctx.n = n
        ctx.k = k
        ctx.alpha = alpha
        
        return out
        
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        u = ctx.u
        l = ctx.l
        n = ctx.n
        k = ctx.k
        alpha = ctx.alpha
        
        device = x.device
        grad_out = torch.zeros(x.shape).to(device)
        
        for j in range(len(n)):
            if j == 0:
                l1 = l
            else:
                l1 = u[j-1].to(device)
                
            u1 = u[j].to(device)
            n1 = n[j]

            if n == 1:
                step = 0
            else:
                step = (u1 - l1)/(n1 - 1)
            quant_width = step
            start_x = l1 + step/2

            i = torch.arange(n1-1).reshape(n1-1,1).unsqueeze(2).to(device)
            
            grad_out = grad_out + torch.sum(k*step*torch.sigmoid(k*(x - start_x - i*quant_width))*(1 - torch.sigmoid(k*(x - start_x - i*quant_width))),dim=0)
            
        x0 = torch.clone(x.detach())
        u0 = torch.clone(u[-1].detach()).item()
        grad_mask = ((x0 < l)|(x0 > u0)) * alpha
        
        grad_out = grad_out + grad_mask
        
        return grad_out, None, None, None, None, None, None

class mlsq(nn.Module):
    def __init__(self, n, l, u, learn_u = False, alpha = 0.01):
        '''
            Multi-level sigmoid quantization function
            
            args:
                n : number of quantization levels for each range; list or int
                l : lower bound for quantization; float
                u : upper bound for each range; list or float
                learn_u : learnable upper bounds; list or True/ False
                alpha : gradient for out of quantization range
        '''
        super(mlsq, self).__init__()
        
        self.n = [n] if isinstance(n, int) else n
        self.l = l
        uu = u if isinstance(u, list) else [u]
        self.learn_u = learn_u if isinstance(learn_u, list) else [learn_u]
        self.u = nn.ParameterList()
        
        for i in range(len(uu)):
            if self.learn_u[i]:
                self.u.append(nn.Parameter(torch.tensor([uu[i]]), requires_grad=True))
            else:
                self.u.append(nn.Parameter(torch.tensor([uu[i]]), requires_grad=False))
        
        self.alpha = alpha
        self.n_ranges = len(self.n)
        self.k = nn.Parameter(data=torch.tensor([0.05], dtype=torch.float32), requires_grad=True)
        # self.q_func = mlsq_func.apply
        
    def multi_level_sigmoid(self, x, slope, j, device, train = True, noise = 0):
        
        if j == 0:
            l = self.l #+ noise
        else:
            l = self.u[j-1].to(device) #+ noise
        u = self.u[j].to(device) #+ noise
        n = self.n[j]
        
        if n == 1:
            step = (u - l)  #+ noise
        else:
            step = (u - l)/(n - 1)  #+ noise
        quant_width = step
        start_x = l + step/2
        
        i = torch.arange(n-1).reshape(n-1,1).unsqueeze(2).to(device)
        if train:
            out = torch.sum(step*torch.sigmoid(slope*(x - start_x - i*quant_width)),dim=0)
        else:
            out = torch.sum(step*torch.round(torch.sigmoid(slope*(x - start_x - i*quant_width))),dim=0)

        return out
    
    def forward(self, xx, k, train = True, noise = 0):
        '''
            forward function
            
            args:
                xx    : input parameters (weights)
                k     : quantization temperature
                train : train loop or validation loop
                noise : noise for the phase step size (upper bounds & lower bounds)
            returns:
                out   : soft/fully quantized paramters
                u     : list of upper bounds
        '''
        
        device = xx.device
        
        if not torch.is_tensor(k):
            k = torch.tensor([k]).to(device)
        
        x0 = torch.clone(xx.detach())
        u0 = torch.clone(self.u[-1].detach()).item()
        mask = ((x0 < self.l) | (x0 > u0)) * 1
        
        out = torch.zeros(xx.shape).to(device)
        for i in range(self.n_ranges):
            out = out + self.multi_level_sigmoid(xx, k, i, device, train, noise)
            
        if train:
            out = out + mask * self.alpha * xx + self.l + noise
        else:
            out = out + self.l
        
        # if k >= 20:
        #     alpha = self.alpha
        # else:
        #     alpha = 0
        # out = self.q_func(xx, k, self.n, self.l, self.u, alpha, train)
            
        return out

    
class dsq(nn.Module):
    def __init__(self, n, l, u, learn_u = False, alpha = 0.2):
        '''
            Differentiable soft quantization function
            
            args:
                n : number of quantization levels for each range; list or int
                l : lower bound for quantization; float
                u : upper bound for each range; list or float
                learn_u : learnable upper bounds; list or True/ False
                alpha: initialization for temperature
        '''
        super(dsq, self).__init__()
        
        self.n = [n] if isinstance(n, int) else n
        self.l = l
        uu = u if isinstance(u, list) else [u]
        self.learn_u = learn_u if isinstance(learn_u, list) else [learn_u]
        self.u = nn.ParameterList()
        self.alpha = nn.Parameter(data=torch.tensor([alpha]), requires_grad=True)
        
        for i in range(len(uu)):
            if self.learn_u[i]:
                self.u.append(nn.Parameter(torch.tensor([uu[i]]), requires_grad=True))
            else:
                self.u.append(nn.Parameter(torch.tensor([uu[i]]), requires_grad=False))
        
        self.n_ranges = len(self.n)
        
    def diff_soft_quant_func(self, x, k, j, device, train = True, noise = 0):
        
        if j == 0:
            l = self.l + noise
            ll = l
        else:
            l = self.u[j-1].to(device) + noise
            ll = noise
        u = self.u[j].to(device) + noise
        n = self.n[j]
        
        if n == 1:
            delta = 0  + noise
        else:
            delta = (u - l)/(n - 1)  + noise
        
        xc = ((x > u) * u) + ((x < l) * l) + (((x >= l) & (x <= u)) * x)
    
        i = (n-1) * (x > u + 0.5*delta) + sum([((x > l + (0.5+p)*delta) & (x <= l + (1.5+p)*delta)) * (p+1) for p in range(n-1)])
        mi = l + (i + 0.5) * delta

        x_sq = (1/(1-k)) * torch.tanh(torch.log(2/k - 1) / delta * (xc - mi))

        if not train:
            x_sq = torch.sgn(x_sq)

        x_hat = ll + delta * (i + (x_sq + 1) / 2)

        return x_hat
    
    def forward(self, xx, k, train = True, noise = 0):
        '''
            forward function
            
            args:
                xx    : input parameters (weights)
                k     : quantization temperature
                train : train loop or validation loop
                noise : noise for the phase step size (upper bounds & lower bounds)
            returns:
                out   : soft/fully quantized paramters
        '''
        
        device = xx.device
        if not torch.is_tensor(k):
            k = torch.tensor([k]).to(device)
        
        out = torch.zeros(xx.shape).to(device)
        for i in range(self.n_ranges):
            out = out + self.diff_soft_quant_func(xx, k, i, device, train, noise)
            
        return out
    
    

def generate_characteristic_curve(quant_func, n, l, u, learn_u, alpha, k, device, cfg):
    '''
        Generate characteristic curve for quantization function and generates plots

        args:
            n : number of quantization levels for each range; list or int
            l : lower bound for quantization; float
            u : upper bound for each range; list or float
            learn_u : learnable upper bounds; list or True/ False
            alpha : gradient for out of quantization range
            k : quantization temperature
            device : device to run on
            cfg: experiment configs dictionary
    '''
    
    if quant_func == 'dsq':
        model = dsq(n, l, u, learn_u = learn_u).to(device)
        if not torch.is_tensor(k):
            k = torch.tensor([k]).to(device)
        k = k.detach()
        # dsq_factor = cfg['dsq_factor'] if 'dsq_factor' in cfg.keys() else 4
        # k = np.exp(-1*(k+1)/dsq_factor)
    else:
        model = mlsq(n, l, u, learn_u = learn_u, alpha = alpha).to(device)
        if not torch.is_tensor(k):
            k = torch.tensor([k]).to(device)
        k = k.detach()
        
    x  = torch.arange(l-3, int(max(u))+3, 0.01, requires_grad=True, dtype = torch.float32).to(device) ## Dummy input
    x.retain_grad()

    y_train = model(x, k, True)
    y_val   = model(x, k, False)

    if quant_func == 'dsq':
        y_train.backward(torch.ones((x.shape[0])).to(device))
    else:
        y_train.backward(torch.ones((1,x.shape[0])).to(device))

    fig1 = go.Figure()
    fig2 = go.Figure()
    

    fig1.add_trace(go.Scatter(x=x.detach().cpu().numpy()/np.pi, y = y_train.detach().cpu().numpy().reshape(x.shape[0],)/np.pi,
                    mode='lines',
                    name='transformation'))
    
    fig1.add_trace(go.Scatter(x=x.detach().cpu().numpy()/np.pi, y = y_val.detach().cpu().numpy().reshape(x.shape[0],)/np.pi,
                    mode='lines',
                    name='val transformation'))
    
    fig2.add_trace(go.Scatter(x=x.detach().cpu().numpy()/np.pi, y = (x.grad).detach().cpu().numpy(),
                    mode='lines',
                    name='derivative'))
    
    fig_all = go.Figure(data=fig1.data + fig2.data)


    return fig1, fig2, fig_all


class fake_quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, N, min_x, max_x):
        min_q = 0
        max_q = N - 1
        
        # compute the scale and the zero point
        s = (max_x - min_x) / (max_q - min_q)
        z = -((min_x / s) - min_q)
        
        # quantization
        x_q = torch.clamp(torch.round(x / s + z), min_q, max_q)
        # dequantization
        x_ = (x_q - z) * s
        
        ctx.save_for_backward(x)
        
        return x_
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_in = torch.ones(grad_output.shape).to(grad_output.device) * grad_output
        
        return grad_in, None, None, None