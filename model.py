import torch
from torch import nn
import torch.functional as F
import os
import numpy as np
from utility import *
import pandas as pd
from kornia.filters import filter2D

def save_models(g, d, location):
    folder = create_folder("SavedModels", location)
    path_to_save = os.path.join("SavedModels", folder)
    print("Saving model to %s" % (path_to_save))
    
    gen_state = g.state_dict()
    torch.save(gen_state, os.path.join(path_to_save, "model.generator"))

    discrim_state = d.state_dict()
    torch.save(discrim_state, os.path.join(path_to_save, "model.discriminator"))

def load_models(folder, device):
    g = generator(device)
    d = discriminator(device)

    gen_params = torch.load(os.path.join(folder, "model.generator"),
    map_location=device)
    discrim_params = torch.load(os.path.join(folder, "model.discriminator"),
    map_location=device)
    
    g.load_state_dict(gen_params)
    d.load_state_dict(discrim_params)

    return g, d

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

def feature_distance(img1, img2, device):
    if(features_model is None):
        model = models.vgg19(pretrained=True).to(device=device)
        model.eval()
        layer = model.features

    if(img1.shape[1] == 1):
        img1 = torch.repeat(img1, 3, axis=1)    
    if(img2.shape[1] == 1):
        img2 = torch.repeat(img2, 3, axis=1)

    img1_feature_vector = layer(img1_tensor)
    img2_feature_vector = layer(img2_tensor)
    
    return ((img1_feature_vector - img2_feature_vector) ** 2).mean()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
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
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512*k)

        # In 32*kx4x4, out 32*kx8x8
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
        self.toHeight1 = nn.Conv2d(32*k, 1, kernel_size=1,
        stride=1, padding=0)

        # In 32*kx8x8, out 16*kx16x16
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
        self.toHeight2 = nn.Conv2d(16*k, 1, kernel_size=1,
        stride=1, padding=0)

        # In 16*kx16x16, out 16*kx32x32
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
        self.toHeight3 = nn.Conv2d(16*k, 1, kernel_size=1,
        stride=1, padding=0)

        # In 16*kx32x32, out 8*kx64x64 
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
        self.toHeight4 = nn.Conv2d(8*k, 1, kernel_size=1,
        stride=1, padding=0)

        # In 8*kx64x64, out 4*kx128x128
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
        self.toHeight5 = nn.Conv2d(4*k, 1, kernel_size=1,
        stride=1, padding=0)

        # In 4*kx128x128, out 2*kx256x256
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
        self.toHeight6 = nn.Conv2d(2*k, 1, kernel_size=1,
        stride=1, padding=0)

        # In 2*kx256x256, out 1*kx512x512
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
        self.toHeight7 = nn.Conv2d(1*k, 1, kernel_size=1,
        stride=1, padding=0)
        
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = x.reshape(x.shape[0], 32*self.k, 4, 4)

        x = self.convBlock1(x + \
        torch.randn(x.shape,device=self.device))
        heightmap = self.toHeight1(x)

        x = self.convBlock2(x + \
        torch.randn(x.shape,device=self.device))
        heightmap = self.upscale(heightmap) + self.toHeight2(x)

        x = self.convBlock3(x + \
        torch.randn(x.shape,device=self.device))
        heightmap = self.upscale(heightmap) + self.toHeight3(x)

        x = self.convBlock4(x + \
        torch.randn(x.shape,device=self.device))
        heightmap = self.upscale(heightmap) + self.toHeight4(x)

        x = self.convBlock5(x + \
        torch.randn(x.shape,device=self.device))
        heightmap = self.upscale(heightmap) + self.toHeight5(x)
        
        x = self.convBlock6(x + \
        torch.randn(x.shape,device=self.device))
        heightmap = self.upscale(heightmap) + self.toHeight6(x)

        x = self.convBlock7(x + \
        torch.randn(x.shape,device=self.device))
        heightmap = self.upscale(heightmap) + self.toHeight7(x)

        heightmap = self.activation(heightmap)
    
        return heightmap

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2D(x, f, normalized=True)

class discriminator(nn.Module):
    def __init__ (self, device, k=32):
        super(discriminator, self).__init__()
        self.device = device
        self.downscale = nn.Upsample(scale_factor=0.5,
        mode="bilinear",align_corners=False)

        # input, 512x512 single channel heightmap
        # Output, kx512x512
        self.fromHeight = nn.Conv2d(1, k, kernel_size=1, 
        stride=1,padding=0)

        # in kx512x512 out kx256x256
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(k, k, kernel_size=3, stride=1, 
            padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(k, k, kernel_size=3, stride=1, 
            padding=1),
            Blur(),
            nn.Conv2d(k, k, kernel_size=3, stride=2, 
            padding=1),
        )

        # in kx256x256 out kx128x128
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(k, k, kernel_size=3, stride=1, 
            padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(k, k, kernel_size=3, stride=1, 
            padding=1),
            Blur(),
            nn.Conv2d(k, k, kernel_size=3, stride=2, 
            padding=1),
        )

        # in kx128x128 out kx64x64
        self.convBlock3 = nn.Sequential(
            nn.Conv2d(k,k, kernel_size=3, stride=1, 
            padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(k,k, kernel_size=3, stride=1, 
            padding=1),
            Blur(),
            nn.Conv2d(k,k, kernel_size=3, stride=2, 
            padding=1),
        )

        # in kx64x64 out kx32x32
        self.convBlock4 = nn.Sequential(
            nn.Conv2d(k, k, kernel_size=3, stride=1, 
            padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(k, k, kernel_size=3, stride=1, 
            padding=1),
            Blur(),
            nn.Conv2d(k, k, kernel_size=3, stride=2, 
            padding=1),
        )

        # in kx32x32 out kx16x16
        self.convBlock5 = nn.Sequential(
            nn.Conv2d(k, k, kernel_size=3, stride=1, 
            padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(k, k, kernel_size=3, stride=1, 
            padding=1),
            Blur(),
            nn.Conv2d(k, k, kernel_size=3, stride=2, 
            padding=1),
        )

        # in kx16x16 out kx8x8
        self.convBlock6 = nn.Sequential(
            nn.Conv2d(k, k, kernel_size=3, stride=1, 
            padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(k, k, kernel_size=3, stride=1, 
            padding=1),
            Blur(),
            nn.Conv2d(k, k, kernel_size=3, stride=2, 
            padding=1),
        )

        # in kx8x8 out kx4x4
        self.convBlock7 = nn.Sequential(
            nn.Conv2d(k, k, kernel_size=3, stride=1, 
            padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(k, k, kernel_size=3, stride=1, 
            padding=1),
            Blur(),
            nn.Conv2d(k, k, kernel_size=3, stride=2, 
            padding=1),
        )

        # in kx4x4 out 1x2x2
        self.convBlock8 = nn.Sequential(
            nn.Conv2d(k, 1, kernel_size=3, stride=1, 
            padding=0),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.fromHeight(x)

        res = self.convBlock1(x)
        x = self.downscale(x) + res

        res = self.convBlock2(x)
        x = self.downscale(x) + res

        res = self.convBlock3(x)
        x = self.downscale(x) + res

        res = self.convBlock4(x)
        x = self.downscale(x) + res

        res = self.convBlock5(x)
        x = self.downscale(x) + res

        res = self.convBlock6(x)
        x = self.downscale(x) + res

        res = self.convBlock7(x)
        x = self.downscale(x) + res

        x = self.convBlock8(x)

        return x.mean()

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
