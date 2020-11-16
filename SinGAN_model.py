import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
from utility import *
import pandas as pd
from kornia.filters import filter2D
import imageio
import math

def save_models(gs, ds, location):
    folder = create_folder("SavedModels", location)
    path_to_save = os.path.join("SavedModels", folder)
    print("Saving model to %s" % (path_to_save))
    
    optimal_noises = {}
    gen_states = {}
    
    for i in range(len(gs)):
        gen_states[str(i)] = gs[i].state_dict()
    torch.save(gen_states, os.path.join(path_to_save, "SinGAN.generators"))

    discrim_states = {}
    for i in range(len(ds)):
        discrim_states[str(i)] = ds[i].state_dict()
    torch.save(discrim_states, os.path.join(path_to_save, "SinGAN.discriminators"))

def load_models(gs, ds, folder, device):
    
    gen_params = torch.load(os.path.join(folder, "SinGAN.generators"),
    map_location=device)
    discrim_params = torch.load(os.path.join(folder, "SinGAN.discriminators"),
    map_location=device)
    for i in range(len(gs)):
        if(str(i) in gen_params.keys()):
            gen_params_compat = OrderedDict()
            gs[i].load_state_dict(gen_params_compat)
            gs[i].to(device)

        if(str(i) in discrim_params.keys()):
            discrim_params_compat = OrderedDict()
            ds[i].load_state_dict(discrim_params_compat)
    return gs, ds

def laplace_pyramid_downscale2D(frame, level, downscale_per_level, device):
    kernel_size = 5
    sigma = 2 * (1 / downscale_per_level) / 6

    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = torch.transpose(x_grid, 0, 1)
    
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                  torch.exp(
                      -torch.sum((xy_grid - mean)**2., dim=-1) /\
                      (2*variance)
                  )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).to(device)
    gaussian_kernel = gaussian_kernel.repeat(frame.shape[1], 1, 1, 1)
    input_size = np.array(list(frame.shape[2:]))
    with torch.no_grad():
        for i in range(level):
            s = (input_size * (downscale_per_level**(i+1))).astype(int)
            frame = F.conv2d(frame, gaussian_kernel, groups=frame.shape[1])
            frame = F.interpolate(frame, size = list(s), mode='nearest')
    del gaussian_kernel
    return frame

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

def generate(generators, batch_size, device):
    with torch.no_grad():
        generated_image = torch.zeros(batch_size, 1,
        generators[0].resolution[0], generators[0].resolution[1]).to(device)
        
        for i in range(0, len(generators)):
            generated_image = F.interpolate(generated_image, 
            size=generators[i].resolution, mode="bilinear", 
            align_corners=False)
            
            generated_image = generators[i](generated_image)   

    return generated_image

def init_scales(dataset, device, min_dim_size = 25, downscale_ratio = 0.75):
    n = round(math.log(min_dim_size / \
    dataset.resolution[0]) / math.log(downscale_ratio))+1
   
    gs = []
    ds = []
    print("The model will have %i scales" % (n))
    for i in range(n):
        scaling = []
        factor =  downscale_ratio**(n - i - 1)
        for j in range(len(dataset.resolution)):
            x = int(dataset.resolution[j] * factor)
            scaling.append(x)
        print("Scale %i: %s" % (i, str(scaling)))
        num_kernels = int((2 ** (5 + (i / 4))) / 3)
        g = SinGAN_Generator(num_kernels, scaling, device).to(device)
        g.apply(weights_init)
        d = SinGAN_Discriminator(num_kernels, device).to(device)
        d.apply(weights_init)
        gs.append(g)
        ds.append(d)
    return n, gs, ds

class SinGAN_Generator(nn.Module):
    def __init__ (self, num_kernels, resolution, device):
        super(SinGAN_Generator, self).__init__()
        self.resolution = resolution
        self.num_kernels = num_kernels
        self.device = device

        modules = []
        for i in range(5):
            # The head goes from 1 channels to num_kernels
            if i == 0:
                modules.append(nn.Sequential(
                    nn.Conv2d(1, num_kernels, kernel_size=3, 
                    stride=1, padding=0),
                    nn.BatchNorm2d(num_kernels),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            # The tail will go from kernel_size to num_channels before tanh [-1,1]
            elif i == 4:  
                tail = nn.Sequential(
                    nn.Conv2d(num_kernels, 1, kernel_size=3, 
                    stride=1, padding=0),
                    nn.Tanh()
                )              
                modules.append(tail)
            # Other layers will have 32 channels for the 32 kernels
            else:
                modules.append(nn.Sequential(
                    nn.Conv2d(num_kernels, num_kernels, kernel_size=3,
                    stride=1, padding=0),
                    nn.BatchNorm2d(num_kernels),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
        self.model = nn.Sequential(*modules)
        
    def forward(self, data, noise=None):
        
        data_padded = F.pad(data, [5, 5, 5, 5])
        noise = torch.randn(data_padded.shape, device=self.device)
        
        noisePlusData = data_padded + noise
        
        output = self.model(noisePlusData)
        return output + data

class SinGAN_Discriminator(nn.Module):
    def __init__ (self, num_kernels, device):
        super(SinGAN_Discriminator, self).__init__()
        self.device=device
        self.num_kernels = num_kernels

        modules = []
        for i in range(5):
            # The head goes from 3 channels (RGB) to num_kernels
            if i == 0:
                modules.append(nn.Sequential(
                    nn.Conv2d(1, num_kernels, 
                    kernel_size=3, stride=1),
                    nn.BatchNorm2d(num_kernels),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            # The tail will go from num_kernels to 1 channel for discriminator optimization
            elif i == 4:  
                tail = nn.Sequential(
                    nn.Conv2d(num_kernels, 1, 
                    kernel_size=3, stride=1)
                )
                modules.append(tail)
            # Other layers will have 32 channels for the 32 kernels
            else:
                modules.append(nn.Sequential(
                    nn.Conv2d(num_kernels, num_kernels, 
                    kernel_size=3, stride=1),
                    nn.BatchNorm2d(num_kernels),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
        self.model =  nn.Sequential(*modules)
        self.model = self.model.to(device)

    def forward(self, x):
        return self.model(x)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_location):
        self.dataset_location = dataset_location
        self.items = os.listdir(dataset_location)        
        self.resolution = imageio.imread(os.path.join(self.dataset_location, 
        self.items[0])).astype(np.float32).shape

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        data = imageio.imread(os.path.join(self.dataset_location, 
        self.items[index])).astype(np.float32)
        data = np2torch(data, "cpu")
        data *= (2.0/255.0)
        data -= 1
        return data.unsqueeze(0)
