import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
import os
import numpy as np
from pytorch_gan_metrics import get_inception_score_and_fid
from cleanfid import fid
from pathlib import Path
from tqdm import tqdm

from models import Q_DCGAN, V_DCGAN
from utils import generate_imgs, weights_init, prepare_for_torchmetric_FID
from losses import QLOSS, VLOSS
from torchvision.transforms.functional import to_pil_image
from torchmetrics.image.fid import FrechetInceptionDistance
import pdb
from fid_calc import clean_fid_compute_fid, pgm_calc_fid

def train_vanilla(params, train_loader, V_net, Q_net, V_optimizer, Q_optimizer, device, log_PATH):

    modelName = "f-GAN-"+params['div']+'-'+params['model']
    div_flag = params['div'] != 'None'
    fake_imgs_save_path = Path(f"./data/{params['dataset']}/fake_imgs")
    fake_imgs_save_path.mkdir(parents=True, exist_ok=True)
    real_imgs_save_path = Path(f"./data/{params['dataset']}/imgs")
    
    if params['div'] == "None":
        Q_criterion = nn.BCEWithLogitsLoss()
        V_criterion = nn.BCEWithLogitsLoss()
        
    else:
        Q_criterion = QLOSS(divergence=params['div'])
        V_criterion = VLOSS(divergence=params['div'])
        
    
    img_list = []
    Q_losses = []
    V_losses = []
    fid_stats = Path.cwd() / "fid_stats" / params['eval_stats_name']

    iter_per_plot = 250

    transform_PIL=transforms.ToPILImage()
    fixed_noise = torch.randn((params['bsize'], params['nz'], 1, 1), device=device)

    for ep in range(params['nepochs']):
        for i, (data, _) in enumerate(train_loader):
            #import pdb; pdb.set_trace()
            b_size=data.shape[0]
            input_data = data.to(device)

            #Train V
            V_net.zero_grad()
            z = torch.randn((b_size, params['nz'], 1, 1), device=device)
            
            input_fake = Q_net(z)
            
            v_real = V_net(input_data).reshape(-1, 1)
            v_fake = V_net(input_fake.detach()).reshape(-1, 1)
            
            if div_flag:
                loss_real = V_criterion(v_real)
                loss_fake = V_criterion(v_fake)
            else:
                real_labels = torch.ones(b_size, 1).to(device)
                fake_labels = torch.zeros(b_size, 1).to(device)
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
                    #fid_fake = (fid_fake + 1) / 2
                    grid = vutils.make_grid(torch.reshape(fid_fake,(b_size,params['nc'],params['imsize'],params['imsize']))[:64], padding=2,
                                                     normalize=True)
                    transform_PIL(grid).save(os.path.join(log_PATH,str(ep)+modelName+"_Last.png"))
        
        
        # print(
        #     f"[INFO] Test run; 5000 images. Calculating FID metrics using pytorch_gan_metrics;\
        #     This method uses calculated stats 64x64 resized original images"
        #     )
        # fid_imgs = generate_imgs_pytorch_gan_metrics(Q_net, device=device, size=50000, z_dim=params['nz'], batch_size=params['bsize'])
        # (IS, IS_std), FID_pgm = get_inception_score_and_fid(
        #      images=fid_imgs[:5000], 
        #      fid_stats_path=str(fid_stats)
        #      )

        # print(f"[INFO] Test run; 5000 images. Calculating FID metrics using clean-fid;\
        #     This method downsizes the generated samples to 32x32 using PIL, then compares (real vs fake folders) or\
        #     (real_stats calculated by clean-fid for 32x32 size vs repo of fake imgs downsized to 32x32 by PIL ")

        # for i, img in enumerate(tqdm(fid_imgs[:5000])):
        #     pil_img = transform_PIL(img).resize((32, 32)) #hard coded to cifar!!!
            
        #     pil_img.save(fake_imgs_save_path / f"{i}.png")

        # #cl_fid_folders = fid.compute_fid(str(fake_imgs_save_path), str(real_imgs_save_path))
        # cl_fid_stats = fid.compute_fid(str(fake_imgs_save_path), dataset_name="clean_fid_cifar_32x32stats", mode='clean', dataset_split='custom')

        # print(f"Pytorch_gan-metrics FID Score: {FID_pgm}")
        # print(f"Clean-fid FID Score from saved_stats: {cl_fid_stats}")   


        
    return Q_net, V_net, V_losses, Q_losses

def train_vanilla(params, train_loader, V_net, Q_net, V_optimizer, Q_optimizer, device, log_PATH):
    modelName = "f-GAN-"+params['div']+'-'+params['model']
    div_flag = params['div'] != 'None'
    iter_per_plot = 250
    if params['fid_method'] == "torchmetrics":
        FID = FrechetInceptionDistance(reset_real_features=False, normalize=False).to(device)

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
            #import pdb; pdb.set_trace()
            b_size=data.shape[0]
            input_data = data.to(device)
            Q_net.train
            
            #Train V
            V_net.zero_grad()
            z = torch.randn((b_size, params['nz'], 1, 1), device=device)
            
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


    ################ 
    # FID EVALUATION
    ################
    if ep % 5 == 0:
        fid_generated_imgs = generate_imgs(
            net_G=Q_net,
            device=device,
            params=params,
            size=50000,
            batch_size=128
            )
        
        #if params['fid_method'] == "cleanfid":
        cleanfid_fid_score = clean_fid_compute_fid(fid_generated_imgs, params, img_size=64)
        #elif params['fid_method'] == "pgm":
        pgm_fid_score = pgm_calc_fid(
                fid_generated_imgs, 
                make_between_01=True,
                params=params)

        #elif params['fid_method'] == 'torchmetrics':
        if ep == 0:
            print("INFO: Calculating torchmetrics real dataset FID statistics")
            for fid_data, _ in tqdm(train_loader):
                fid_data = fid_data.to(device)
                fid_input_data_real = prepare_for_torchmetric_FID(fid_data, between_01=True).to(device)
                FID.update(fid_input_data_real, real=True)
        else:
            print("INFO: Calculating torchmetrics generated dataset statistics")
            fid_fake_data = prepare_for_torchmetric_FID(fid_generated_imgs, make_between_01=True).to(device)
            FID.update(fid_fake_data, real=False)
            torchmetrics_fid_score = FID.compute()
            FID.reset()
        print(f"[INFO]: cleanfid score: {cleanfid_fid_score}")
        print(f"[INFO]: pgm score: {pgm_fid_score}")
        print(f"[INFO]: torchmetrics score: {torchmetrics_fid_score}")


            

            


    print(f"[INFO] Epoch {ep} FID score: {fid_score}" )

       

def train_vanilla_with_torchmetrics(params, train_loader, V_net, Q_net, V_optimizer, Q_optimizer, device, log_PATH):

    modelName = "f-GAN-"+params['div']+'-'+params['model']
    div_flag = params['div'] != 'None'
    fake_imgs_save_path = Path(f"./data/{params['dataset']}/fake_imgs")
    fake_imgs_save_path.mkdir(parents=True, exist_ok=True)
    real_imgs_save_path = Path(f"./data/{params['dataset']}/imgs")
    FID = FrechetInceptionDistance(reset_real_features=False, normalize=True).to(device=device)
    
    if params['div'] == "None":
        Q_criterion = nn.BCELoss()
        V_criterion = nn.BCELoss()
        
    else:
        Q_criterion = QLOSS(divergence=params['div'])
        V_criterion = VLOSS(divergence=params['div'])
        
    
    img_list = []
    Q_losses = []
    V_losses = []

    iter_per_plot = 250

    transform_PIL=transforms.ToPILImage()
    fixed_noise = torch.randn((params['bsize'], params['nz'], 1, 1), device=device)

    for ep in range(params['nepochs']):
        for i, (data, _) in enumerate(train_loader):
            #import pdb; pdb.set_trace()
            b_size=data.shape[0]
            input_data = data.to(device)
            Q_net.train
            
            #Train V
            V_net.zero_grad()
            z = torch.randn((b_size, params['nz'], 1, 1), device=device)
            
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

            # testing if this makes subsequent epochs faster and same results
            if ep == 0:
                print("INFO] Computing Statistics on real dataset")
                fid_input_data_real = prepare_for_torchmetric_FID(input_data, between_01=True).to(device)
                FID.update(fid_input_data_real, real=True)
                print("INFO[ Statistics calculation complete")
            fid_input_data_fake = prepare_for_torchmetric_FID(input_fake, between_01=True).to(device)
            FID.update(fid_input_data_fake, real=False)

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

        epoch_fid = FID.compute()
        print(f"Torchmetrics FID Score: {epoch_fid.item()}")
        FID.reset()