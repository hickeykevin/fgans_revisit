import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

from models import Q_DCGAN, V_DCGAN
from utils import generate_imgs, weights_init
from losses import QLOSS, VLOSS
from torchvision.transforms.functional import to_pil_image


def train_vanilla(params, train_loader, V_net, Q_net, V_optimizer, Q_optimizer, device, log_PATH):
    modelName = "f-GAN-"+params['div']+'-'+params['model']
    div_flag = params['div'] != 'None'
    iter_per_plot = 250

    if params['div'] == "None":
        Q_criterion = nn.BCELoss()
        V_criterion = nn.BCELoss()
        
    else:
        Q_criterion = QLOSS(divergence=params['div'])
        V_criterion = VLOSS(divergence=params['div'])
        
    Q_losses = []
    V_losses = []
    transform_PIL=transforms.ToPILImage()
    fixed_noise = torch.randn(params['bsize'], *params['nz'], device=device)

    for ep in range(params['nepochs']):
        for i, (data, _) in enumerate(train_loader):
            b_size=data.shape[0]
            input_data = data.to(device)
            Q_net.train
            
            #Train V
            V_net.zero_grad()
            z = torch.randn((b_size, *params['nz']), device=device)
            
            input_fake = Q_net(z)
            
            v_real = V_net(input_data).reshape(-1, 1)
            v_fake = V_net(input_fake.detach()).reshape(-1, 1)
            
            
            if div_flag:
                loss_real = V_criterion(v_real)
                loss_fake = V_criterion(v_fake)
            else:
                real_labels = torch.ones((b_size, 1), device=device)
                fake_labels = torch.zeros((b_size, 1), device=device)
                loss_real = V_criterion(v_real, real_labels)
                loss_fake = V_criterion(v_fake, fake_labels)

            loss_V = loss_real + loss_fake
            loss_real.backward(retain_graph=True)
            loss_fake.backward()

            V_optimizer.step()

            #Train G 
            Q_net.zero_grad()
            v_fake = V_net(input_fake).reshape(-1, 1)
            if div_flag:      
                loss_Q = Q_criterion(v_fake)
            else:
                loss_Q = Q_criterion(v_fake, real_labels)
            
            loss_Q.backward()
            Q_optimizer.step()

            

            if (i+1)%iter_per_plot == 0 or i ==0:
                print(f"Epoch {ep}/{params['nepochs']}, Step {i+1}/{len(train_loader)}, V_loss: {loss_V.item():.4f}, Q_loss: {loss_Q.item():.4f}")
                Q_losses.append(loss_Q.item())
                V_losses.append(loss_V.item())

                with torch.no_grad():
                    torch.save(Q_net.state_dict(),log_PATH + '/' + "Q_"+modelName+ "_"+ str(ep)+"_.pth")
                    torch.save(V_net.state_dict(),log_PATH + '/' + "V_"+modelName+ "_"+ str(ep)+"_.pth") 
                    # saving imgs for enuring GAN is learning something
                    fid_fake = Q_net(fixed_noise).detach().cpu()
                    grid = vutils.make_grid(torch.reshape(fid_fake,(b_size,params['nc'],params['imsize'],params['imsize']))[:64], padding=2,
                                                     normalize=True)
                    transform_PIL(grid).save(os.path.join(log_PATH,str(ep)+modelName+"_Last.png"))


    # ################ 
    # # FID EVALUATION
    # ################
    #     if ep % 1 == 0:
    #         fid_generated_imgs = generate_imgs(
    #             net_G=Q_net,
    #             device=device,
    #             params=params,
    #             size=50000,
    #             batch_size=128
    #             )
            
            


            

            



       

