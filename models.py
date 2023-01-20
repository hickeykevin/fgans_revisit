#TODO: Fix naming of layers for the mc decoders (change name to not conflict with diff decoder layers)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_DCGAN(nn.Module):  
    def __init__(self, params):
        super().__init__()
        if params['imsize'] == 32:
            final_layer_kernel_size = 1
            final_layer_stride_size = 1
            final_layer_padding_size = 0
        elif params['imsize'] == 64:
            final_layer_kernel_size = 4
            final_layer_stride_size = 2
            final_layer_padding_size = 1

        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(params['nz'][0], params['ngf']*8,
            kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(params['ngf']*8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(params['ngf']*8, params['ngf']*4,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ngf']*4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(params['ngf']*4, params['ngf']*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ngf']*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(params['ngf']*2, params['ngf'],
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ngf'])

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(params['ngf'], params['nc'],
            final_layer_kernel_size, 
            final_layer_stride_size, 
            final_layer_padding_size, 
            bias=False
            )
        #Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        x = torch.tanh(self.tconv5(x)) #need 'trick' to get between [0,1] for FID
        #x = F.tanh(self.tconv5(x))

        return x
    
class Q_DCGAN_128(nn.Module):  
    def __init__(self, params):
        super().__init__()
        if params["imsize"]==64:
            final_layer_kernel_size = 4
            final_layer_stride_size = 2
            final_layer_padding_size =1 

        # Input is the latent vector Z.
        self.tconv0 = nn.ConvTranspose2d(params['nz'][0], params['ngf']*16,
            kernel_size=4, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(params['ngf']*16)
        
        # Input Dimension: (ngf*16) x 4 x 4
        self.tconv1 = nn.ConvTranspose2d(params['ngf']*16, params['ngf']*8,
            kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(params['ngf']*8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(params['ngf']*8, params['ngf']*4,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ngf']*4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(params['ngf']*4, params['ngf']*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ngf']*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(params['ngf']*2, params['ngf'],
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ngf'])

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(
            params['ngf'], params['nc'],
            final_layer_kernel_size, 
            final_layer_stride_size, 
            final_layer_padding_size, 
            bias=False
            )
        #Output Dimension: (nc) x imsize x imsize

    def forward(self, x):
        x = F.relu(self.bn0(self.tconv0(x)))
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        x = F.tanh(self.tconv5(x))

        return x
    

class V_DCGAN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        if params['imsize'] == 32:
            final_layer_kernel_size = 2
            final_layer_stride_size = 1
            final_layer_padding_size = 0
        elif params['imsize'] == 64:
            final_layer_kernel_size = 4
            final_layer_stride_size = 2
            final_layer_padding_size = 1
        
        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(params['nc'], params['ndf'],
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(params['ndf'], params['ndf']*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ndf']*2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(params['ndf']*2, params['ndf']*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ndf']*4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(params['ndf']*4, params['ndf']*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ndf']*8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(
            params['ndf']*8, 1, 
            final_layer_kernel_size, 
            final_layer_stride_size, 
            final_layer_padding_size, 
            bias=False
            )

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)

        if self.params['div'] == "None":
            x = F.sigmoid(self.conv5(x))
        else:
            x = self.conv5(x)
        return x
    
class V_DCGAN_128(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input Dimension: (nc) x 128 x 128
        self.conv1 = nn.Conv2d(params['nc'], params['ndf'],
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 64 x 64
        self.conv2 = nn.Conv2d(params['ndf'], params['ndf']*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ndf']*2)

        # Input Dimension: (ndf*2) x 32 x 32
        self.conv3 = nn.Conv2d(params['ndf']*2, params['ndf']*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ndf']*4)

        # Input Dimension: (ndf*4) x 16 x 16
        self.conv4 = nn.Conv2d(params['ndf']*4, params['ndf']*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ndf']*8)

        # Input Dimension: (ndf*8) x 8 x 8
        self.conv5 = nn.Conv2d(params['ndf']*8, params['ndf']*16,
            4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(params['ndf']*16)
        
        # Input Dimension: (ndf*8) x 4 x 4
        self.conv6 = nn.Conv2d(params['ndf']*16, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2, True)

        x = self.conv6(x)

        return x

        