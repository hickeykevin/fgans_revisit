import torch
import torch.nn as nn

        
class VLOSS(nn.Module):
    def __init__(self,divergence="JSD"):
        super(VLOSS,self).__init__()
        self.activation = Activation_g(divergence)
    def forward(self,v):
        return torch.mean(self.activation(v))
    
class QLOSS(nn.Module):
    def __init__(self,divergence="JSD"):
        super(QLOSS,self).__init__()
        self.conjugate = Conjugate_f(divergence)
        self.activation = Activation_g(divergence)
        self.eps = 1e-4
    def forward(self,v):
        return torch.mean(-self.conjugate(self.activation(v)))
    
class Activation_g(nn.Module):
    def __init__(self,divergence="JSD"):
        super(Activation_g,self).__init__()
        self.divergence =divergence
        self.eps = 1e-4
    def forward(self,v):
        divergence = self.divergence
        if divergence == "KLD":
            return v 
        elif divergence == "RKL":
            return torch.clamp(-torch.exp(-v), min=-torch.exp(torch.tensor(15))) # if v goes -infinity: if v < -large_Valuev[-100] then activation v maybe clamp; check if getting torch.inf values and see if its technically working
        elif divergence == "CHI":
            return v
        elif divergence == "SQH":
            return torch.tensor(1.)-torch.clamp(torch.exp(-v), max=torch.exp(torch.tensor(15))) - torch.tensor(self.eps) #maybe make esp larger
        elif divergence == "JSD":
            v = v.view(v.shape[0], 1)
            if v.is_cuda: 
                v_cat = torch.concat([torch.zeros(v.shape).to('cuda'), v], dim=1)
            else:
                v_cat = torch.concat([torch.zeros(v.shape), v], dim=1)
            return torch.log(torch.tensor(2.)) - torch.logsumexp(-v_cat,1) - self.eps #substitute of 1+e^-v
#             return torch.log(torch.tensor(2.))-torch.log(1.0+torch.exp(-v)) #- self.eps
        elif divergence == "GAN":
            v = v.view(v.shape[0], 1)
            if v.is_cuda:
              v_cat = torch.concat([torch.zeros(v.shape).to('cuda'), v], dim=1)
            else:
              v_cat = torch.concat([torch.zeros(v.shape), v], dim=1)
            return -torch.logsumexp(-v_cat,1) - self.eps # log sigmoid

    
class Conjugate_f(nn.Module):
    def __init__(self,divergence="JSD"):
        super(Conjugate_f,self).__init__()
        self.divergence = divergence
        self.eps = 1e-4
    def forward(self,t):
        divergence= self.divergence
        if divergence == "KLD":
            return torch.clamp(torch.exp(t-1), max=torch.exp(torch.tensor(15)))
        elif divergence == "RKL":
            return torch.tensor(-1) - torch.log(-t + self.eps)
        elif divergence == "CHI":
            return 0.25*t**2+t
        elif divergence == "SQH":
            return t/(torch.tensor(1.)-t)
        elif divergence == "JSD":
            return -torch.log(2.0-torch.exp(t) + self.eps)
        elif divergence == "GAN":
            return  -torch.log(1.0-torch.exp(t))
        
class Conjugate_double_prime(nn.Module):
    def __init__(self,divergence="GAN"):
        super(Conjugate_double_prime,self).__init__()
        self.divergence = divergence
    def forward(self,v):
        divergence= self.divergence
        if divergence == "KLD":
            return torch.exp(v-1)
        elif divergence == "RKL":
            return 1./(v**2)
        elif divergence == "CHI":
            return 0.5*(v**0)
        elif divergence == "SQH":
            return (2*v)/((1-v)**3) + 2/((1-v)**2)
        elif divergence == "JSD":
            return 2*(torch.exp(v))/((2-torch.exp(v))**2)
        elif divergence == "GAN":
            return  (torch.exp(v))/((1-torch.exp(v))**2)