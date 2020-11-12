import torch
from torch import nn
import torch.functional as F
import os
import numpy as np
from utility import *
import pandas as pd

def save_models(g, d, location):
    folder = create_folder("SavedModels", location)
    path_to_save = os.path.join("SavedModels", folder)
    print("Saving model to %s" % (path_to_save))
    
    gen_state = g.state_dict()
    torch.save(gen_state, os.path.join(path_to_save, "model.generator"))

    discrim_state = d.state_dict()
    torch.save(discrim_state, os.path.join(path_to_save, "model.discriminator"))

def load_models(folder, device):
    generators = generator(device)
    discriminators = discriminator(device)

    gen_params = torch.load(os.path.join(folder, "model.generator"),
    map_location=device)
    discrim_params = torch.load(os.path.join(folder, "model.discriminator"),
    map_location=device)
    
    generator.load_state_dict(gen_params)
    discriminator.load_state_dict(discrim_params)

    return generator, discriminator

def calc_gradient_penalty(discrim, real_data, fake_data, device):
    #print real_data.size()
    alpha = torch.rand(1, 1, device=device)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    #interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discrim(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# input parameters: [smooth/jagged, steepness, elevation]
class generator(nn.Module):
    def __init__ (self, device, k=16):
        super(generator, self).__init__()
        self.device = device
        modules = []
        self.k = k
        self.upscale = nn.Upsample(scale_factor=2,mode="bilinear",
        align_corners=False)

        # Input is 16, output is 256
        self.fc1 = nn.Linear(16, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512*k)

        # In 32*kx4x4, out 32*kx8x8
        self.resConv1 = nn.Conv2d(32*k, 32*k, kernel_size=1, 
        stride=1, padding=0)
        self.convBlock1 = nn.Sequential(
            nn.BatchNorm2d(32*k),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2,mode="bilinear",
            align_corners=False),
            nn.Conv2d(32*k, 32*k, kernel_size=3, stride=1, 
            padding=1),
            nn.BatchNorm2d(32*k),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32*k, 32*k, kernel_size=3, stride=1, 
            padding=1)
        )
        # In 32*kx8x8, out 16*kx16x16
        self.resConv2 = nn.Conv2d(32*k, 16*k, kernel_size=1, 
        stride=1, padding=0)
        self.convBlock2 = nn.Sequential(
            nn.BatchNorm2d(32*k),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2,mode="bilinear",
            align_corners=False),
            nn.Conv2d(32*k, 16*k, kernel_size=3, stride=1, 
            padding=1),
            nn.BatchNorm2d(16*k),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16*k, 16*k, kernel_size=3, stride=1, 
            padding=1)
        )
        # In 16*kx16x16, out 16*kx32x32
        self.resConv3 = nn.Conv2d(16*k, 16*k, kernel_size=1, 
        stride=1, padding=0)
        self.convBlock3 = nn.Sequential(
            nn.BatchNorm2d(16*k),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2,mode="bilinear",
            align_corners=False),
            nn.Conv2d(16*k, 16*k, kernel_size=3, stride=1, 
            padding=1),
            nn.BatchNorm2d(16*k),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16*k, 16*k, kernel_size=3, stride=1, 
            padding=1)
        )
        # In 16*kx32x32, out 8*kx64x64 
        self.resConv4 = nn.Conv2d(16*k, 8*k, kernel_size=1, 
        stride=1, padding=0)
        self.convBlock4 = nn.Sequential(
            nn.BatchNorm2d(16*k),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2,mode="bilinear",
            align_corners=False),
            nn.Conv2d(16*k, 8*k, kernel_size=3, stride=1, 
            padding=1),
            nn.BatchNorm2d(8*k),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8*k, 8*k, kernel_size=3, stride=1, 
            padding=1)
        )
        # In 8*kx64x64, out 4*kx128x128
        self.resConv5 = nn.Conv2d(8*k, 4*k, kernel_size=1, 
        stride=1, padding=0)
        self.convBlock5 = nn.Sequential(
            nn.BatchNorm2d(8*k),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2,mode="bilinear",
            align_corners=False),
            nn.Conv2d(8*k, 4*k, kernel_size=3, stride=1, 
            padding=1),
            nn.BatchNorm2d(4*k),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*k, 4*k, kernel_size=3, stride=1, 
            padding=1)
        )
        # In 4*kx128x128, out 2*kx256x256
        self.resConv6 = nn.Conv2d(4*k, 2*k, kernel_size=1, 
        stride=1, padding=0)
        self.convBlock6 = nn.Sequential(
            nn.BatchNorm2d(4*k),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2,mode="bilinear",
            align_corners=False),
            nn.Conv2d(4*k, 2*k, kernel_size=3, stride=1, 
            padding=1),
            nn.BatchNorm2d(2*k),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*k, 2*k, kernel_size=3, stride=1, 
            padding=1)
        )
        # In 2*kx256x256, out 1*kx512x512
        self.resConv7 = nn.Conv2d(2*k, 1*k, kernel_size=1, 
        stride=1, padding=0)
        self.convBlock7 = nn.Sequential(
            nn.BatchNorm2d(2*k),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2,mode="bilinear",
            align_corners=False),
            nn.Conv2d(2*k, 1*k, kernel_size=3, stride=1, 
            padding=1),
            nn.BatchNorm2d(1*k),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1*k, 1*k, kernel_size=3, stride=1, 
            padding=1)
        )
        
        # In 1x512x512, out 1x512x512
        self.convBlock8 = nn.Sequential(
            nn.BatchNorm2d(1*k),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1*k, 1, kernel_size=3, stride=1, 
            padding=1)
        )

        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.reshape(x.shape[0], 32*self.k, 4, 4)

        res = self.convBlock1(x)
        x = res + self.resConv1(self.upscale(x))

        res = self.convBlock2(x + \
        torch.randn(x.shape,device=self.device))
        x = res + self.resConv2(self.upscale(x))

        res = self.convBlock3(x + \
        torch.randn(x.shape,device=self.device))
        x = res + self.resConv3(self.upscale(x))

        res = self.convBlock4(x + \
        torch.randn(x.shape,device=self.device))
        x = res + self.resConv4(self.upscale(x))

        res = self.convBlock5(x + \
        torch.randn(x.shape,device=self.device))
        x = res + self.resConv5(self.upscale(x))
        
        res = self.convBlock6(x + \
        torch.randn(x.shape,device=self.device))
        x = res + self.resConv6(self.upscale(x))

        res = self.convBlock7(x + \
        torch.randn(x.shape,device=self.device))
        x = res + self.resConv7(self.upscale(x))

        x = self.convBlock8(x)
        x = self.activation(x)

        return x

class discriminator(nn.Module):
    def __init__ (self, device):
        super(discriminator, self).__init__()
        self.device = device
        modules = []
        # input, 512x512 single channel heightmap
        modules.append(nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, stride=2, 
            padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        ))
        #output: 254x254 with 128 channels
        modules.append(nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, stride=2, 
            padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        ))
        #output: 125x125 with 64 channels
        modules.append(nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=2, 
            padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        ))
        #output: 60x60 with 32 channels
        modules.append(nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, stride=2, 
            padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        ))
        # output: 28x28 with 16 channels
        modules.append(nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=5, stride=2, 
            padding=0),
            nn.BatchNorm2d(1),
            nn.Tanh()
        ))
        # output: 12x12 with 1 channel
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x).mean()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_location):
        self.dataset_location = dataset_location
        self.items = os.listdir(dataset_location)
        self.items.remove("stats.txt")
        stats = np.array(pd.read_csv( \
        os.path.join(dataset_location, "stats.txt"), 
        sep="\s+", header=None))

        self.min = stats[0, 0]
        self.max = stats[0, 1]
        '''
        self.min = None
        self.max = None
        
        print("Loading dataset...")
        for filename in self.items:
            d = np.load(os.path.join(self.dataset_location, 
            filename))
            if(self.min is None):
                self.min = d.min()
            elif(self.min > d.min()):
                self.min = d.min()
            if(self.max is None):
                self.max = d.max()
            elif(self.max < d.max()):
                self.max = d.max()
        print("Datset loaded. Min val: %0.02f, \
        max val: %0.02f" % (self.min, self.max))
        '''
        

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        data = np.load(os.path.join(self.dataset_location, 
        self.items[index]))
        data = np2torch(data, "cpu")
        '''
        data -= self.min
        data *= 2 / (self.max - self.min)
        data -= 1
        '''
        data -= data.min() - 1
        data *= (2/(data.max()+2))
        data -= 1
        return data.unsqueeze(0)
