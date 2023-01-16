
from dataloaders import CIFAR10DataModule
from models import Q_DCGAN
import torch
import pdb
from dataclasses import asdict, dataclass

@dataclass
class Config:

    dataset: str = 'cifar10'#Dataset. Options: cifar10, mnist, stl10, celeba
    exp_name: str =  'Jan_test_of_FID_using_JSD+DCGAN'#Name of experiment.
    div: str = 'None' #Divergence to use with f-gan. Choices: None,  'JSD', x'SQH', 'GAN, 'KLD', 'RKL', 'CHI', 'Wasserstein'
    model: str =  'DCGAN'#Backbone model. Options: PGC, DCGAN, DCGAN_128, CNN, PGM
    bsize: int =  64  #Batch size during training.
    imsize: int = 64
    nc: int =  3 # Number of channles in the training images. For coloured images this is 3.
    nz: int =  128 # Size of the Z latent vector (the input to the generator).
    ngf: int = 64 # Size of feature maps in the generator. The depth will be multiples of this.
    ndf: int = 64 # Size of features maps in the discriminator. The depth will be multiples of this.
    nepochs: int =  200 # Number of training epochs.
    lr: float = 0.0002 #Learning rate for optimizers
    beta1: float =  0.5 #Beta1 hyperparam for Adam optimizer
    beta2: float =  0.5 #Beta2 hyperparam for Adam optimizer
    disc_per_gen: int =  1  #How many epochs to train discriminator per 1 generator epoch
    nwork: int =  1 #Number of workers
    eval_stats_name: str = 'cifar10_64x64.npz'

import torch
_ = torch.manual_seed(123)
from torchmetrics.image.fid import FrechetInceptionDistance
device = torch.device('cuda:0')
fid = FrechetInceptionDistance(feature=64).to(device=device)
# generate two slightly overlapping image intensity distributions
imgs_dist1 = torch.randint(0, 200, (1000, 3, 299, 299), dtype=torch.uint8, device=device)
imgs_dist2 = torch.randint(100, 255, (1000, 3, 299, 299), dtype=torch.uint8, device=device)
fid.update(imgs_dist1, real=True)
fid.update(imgs_dist2, real=False)
fid.compute()




# overall strategy:
# - keep normalize transformation in transforms.compose
# - allow Gen to output between [-1, 1] (uses tanh currently)
# -- allow Disc to train with real images between [-1, 1] and fake images [-1, 1]
# -- calculating losses as such
# - For FID evaluation:
 
# -- 1.) pytorch_gan_metrics options: 
## -- ALL GENERATED INSTANCES TENSORS BETWEEN [0, 1]
## -- calc stats using image_size=64 option in function, use stats against generated images of 64x64 size
## ---- risky bc its using pytorch resize function for 64x64 resizing, which is frowned upon
## -- use precalculated/or calc stats on original images (size=32), downsample generated images to 32x32 (using PIL), use stats against resized images
## ---- more trust worthy than first option, but unsure how downsizing affects generated imgs

# -- 2.) clean-fid options
## -- SINCE USING FOLDERS, IMAGES WILL TRANSFORMED TOPIL AND SAVED FROM PIL.SAVE FUNCTION
## -- replace all original image files by upsampling to 64x64 using PIL, precompute statistics, save all generated instances to folder, use compute_fid with folderfake and 64x64precomputed stats
## ---- note: need to remove resizing transforms from pytorch
## -- use/calc stats for 32x32 size images, downsample all generated instances to 32x32, save to folder, use compute_fid with folder fake and 32x32 stats

# -- 3.)torchmetrics
## -- don't change anything; allow real to be 32x32, fake to be 64x64, but resize each using PIL to be size 299x299 AND ENSURE ALL FID IMAGES ARE BETWEEN [0, 1] WITH NORMALIZE FLAG=TRUE, then feed those into update() function FOR EACH BATCH
## -- at epoch end, run the compute(), which returns that epoch's fid score
