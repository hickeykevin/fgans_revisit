from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:

    dataset: str = 'cifar10'#Dataset. Options: cifar10, mnist, stl10, celeba

    exp_name: str =  '1/16_test'#Name of experiment.
    div: str = 'None' #Divergence to use with f-gan. Choices: None,  'JSD', x'SQH', 'GAN, 'KLD', 'RKL', 'CHI', 'Wasserstein'
    model: str =  'DCGAN'#Backbone model. Options: PGC, DCGAN, DCGAN_128, CNN, PGM
    bsize: int =  64  #Batch size during training.
    imsize: int = 64
    nc: int =  3 # Number of channles in the training images. For coloured images this is 3.
    nz: Tuple =  (128, 1, 1) # Size and dimensionality of the Z latent vector (the input to the generator).
    ngf: int = 64 # Size of feature maps in the generator. The depth will be multiples of this.
    ndf: int = 64 # Size of features maps in the discriminator. The depth will be multiples of this.
    nepochs: int =  200 # Number of training epochs.
    lr: float = 0.0002 #Learning rate for optimizers
    beta1: float =  0.5 #Beta1 hyperparam for Adam optimizer
    beta2: float =  0.5 #Beta2 hyperparam for Adam optimizer
    disc_per_gen: int =  1  #How many epochs to train discriminator per 1 generator epoch
    nwork: int =  1 #Number of workers
    fid_method: str = 'torchmetrics' #choices are torchmetrics, pgm, cleanfid
    eval_stats_name: str = 'cifar10_64x64.npz'
    cleanfid_downsample: bool = True
    