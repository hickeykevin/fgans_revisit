from platform import python_version
python_version()

import os
import torch
import torch.optim as optim
import numpy as np

from models import Q_DCGAN, V_DCGAN, Q_DCGAN_128, V_DCGAN_128
from utils import weights_init
from losses import QLOSS, VLOSS
from dataloaders import CIFAR10DataModule
from train_vanilla import train_vanilla
from config import Config
from dataclasses import asdict
torch.autograd.set_detect_anomaly(True)

config = Config()
params = asdict(config) 

PATH = os.path.join(os.getcwd(),'data' + '/' + config.dataset)
log_PATH_dataset = os.path.join(os.getcwd(), "logs" + '/' + config.dataset)
log_PATH = os.path.join(log_PATH_dataset, config.exp_name)
print(PATH)

if not os.path.exists(PATH):
    os.makedirs(PATH)
    print("The new dataset directory is created!")
    
if not os.path.exists(log_PATH_dataset):
    os.makedirs(log_PATH_dataset)
    print("The new log dataset directory is created!")
    
if not os.path.exists(log_PATH):
    os.makedirs(log_PATH)
    print("The new experiment directory is created!")

modelName = "f-GAN-"+config.div +'-'+ config.model 

TINY = 1e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

manualSeed = 3
print("Random Seed: ",manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

# %%
if config.dataset == 'cifar10':
    train_set, train_loader, test_set, test_loader = CIFAR10DataModule(params, PATH)
    
train_iter = iter(train_loader)
test_iter = iter(test_loader)

if config.model == 'DCGAN':
    print("DCGAN Backbone")
    Q_net = Q_DCGAN(params).to(device)
    V_net = V_DCGAN(params).to(device)
    
elif config.model == 'DCGAN_128':
    print("DCGAN_128 Backbone")
    Q_net = Q_DCGAN_128(params).to(device)
    V_net = V_DCGAN_128(params).to(device)
    
else:
    print('model not defined')
    print('using DCGAN backbone')
    Q_net = Q_DCGAN(params).to(device)
    V_net = V_DCGAN(params).to(device)
    
Q_optimizer = optim.Adam(Q_net.parameters(),lr=config.lr, betas=(config.beta1, config.beta2))
V_optimizer = optim.Adam(V_net.parameters(),lr=config.lr, betas=(config.beta1, config.beta2))

Q_net.apply(weights_init)
V_net.apply(weights_init)

if __name__ == "__main__":
    print('Vanilla training')
    print('Starting Training...')
    
    train_vanilla(params, train_loader, V_net, Q_net, 
                    V_optimizer, Q_optimizer, device, log_PATH)
    



