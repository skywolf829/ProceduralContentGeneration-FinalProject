import torch
from torchvision.utils import save_image, make_grid
from stylegan2_pytorch import ModelLoader, Trainer
import os
from datetime import datetime
import torch.optim as optim
import imageio
import numpy as np
from torchviz import make_dot
import torch.nn as nn
import torch.nn.functional as F
from graphviz import Source

os.environ["PATH"] += os.pathsep + "C:/Users/Sky/miniconda3/Library/bin/graphviz"

def exists(val):
    return val is not None

def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda(device)

def timestamped_filename(prefix = 'generated-'):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'

def noise_to_styles(model, noise, trunc_psi = None):
    w = model.GAN.S(noise)
    if exists(trunc_psi):
        print("truncing")
        w = model.truncate_style(w)
    return w

def styles_to_images(model, w, noise):
    batch_size, *_ = w.shape
    num_layers = model.GAN.G.num_layers
    image_size = model.image_size
    w_def = [(w, num_layers)]

    w_tensors = styles_def_to_tensor(w_def)

    images = model.GAN.G(w_tensors, noise)
    images.clamp_(0., 1.)
    return images

def get_img_from_tensor(t, nrow=8):
    imgs = make_grid(t, nrow=nrow).detach().cpu().numpy().swapaxes(0,1).swapaxes(1,2)
    imgs *= 255 
    imgs = imgs.astype(np.uint8)
    return imgs

def laplacian(imgs, device):
    m = nn.ReplicationPad2d(1)

    weights = torch.tensor(
        np.array([
        [1/8., 1/8., 1/8.], 
        [1/8., -1. , 1/8.],
        [1/8., 1/8., 1/8.]
        ]
    ).astype(np.float32)).to(device)
    weights = weights.view(1, 1, 3, 3)
    imgs = m(imgs)
    output = F.conv2d(imgs, weights)
    return output

def sobel(imgs, axis, device):
    m = nn.ReplicationPad2d(1)
    if(axis == 0):
        weights = torch.tensor(
            np.array([
            [-1/8, 0, 1/8], 
            [-1/4, 0, 1/4],
            [-1/8, 0, 1/8]
            ]
        ).astype(np.float32)).to(device)
        
    elif(axis == 1):
        weights = torch.tensor(
            np.array([
            [-1/8, -1/4, -1/8], 
            [   0,    0,    0], 
            [ 1/8,  1/4,  1/8]
            ]
        ).astype(np.float32)).to(device)
    weights = weights.view(1, 1, 3, 3)
    imgs = m(imgs)
    output = F.conv2d(imgs, weights)
    return output

model_args = dict(
    name = 'default',
    results_dir = './results',
    models_dir = './models',
    batch_size = 8,
    gradient_accumulate_every = 6,
    image_size = 128,
    network_capacity = 4,
    fmap_max = 512,
    transparent = False,
    lr = 2e-4,
    lr_mlp = 0.1,
    ttur_mult = 1.5,
    rel_disc_loss = False,
    num_workers = 16,
    save_every = 1000,
    evaluate_every = 1000,
    trunc_psi = 0.75,
    fp16 = False,
    cl_reg = False,
    fq_layers = [],
    fq_dict_size = 256,
    attn_layers = [],
    no_const = False,
    aug_prob = 0.,
    aug_types = ['translation', 'cutout'],
    top_k_training = False,
    generator_top_k_gamma = 0.99,
    generator_top_k_frac = 0.5,
    dataset_aug_prob = 0.,
    calculate_fid_every = None,
    mixed_prob = 0.9,
    log = False
)

num_image_tiles = 8
load_from = -1
results_dir = './results'
name = 'default'

model = Trainer(**model_args)
model.load(load_from)
model.GAN.train(False)
'''
for params in model.GAN.parameters():
    params.requires_grad = False # Freeze all weights
'''

def smoothness_loss(imgs):
    laps = laplacian(images,"cuda:0")
    laps = torch.abs(laps)
    return laps.mean()

def elevation_loss(imgs):
    return imgs.mean()

def vertical_orientation_loss(imgs):
    sobel_x = torch.abs(sobel(imgs, 1, "cuda:0")).mean()
    sobel_y = torch.abs(sobel(imgs, 0, "cuda:0")).mean()
    return sobel_x / sobel_y

def horizontal_orientation_loss(imgs):
    sobel_x = torch.abs(sobel(imgs, 1, "cuda:0")).mean()
    sobel_y = torch.abs(sobel(imgs, 0, "cuda:0")).mean()
    return sobel_y / sobel_x


noise   = torch.randn(9, 512).cuda()
noise.requires_grad = True

img_noises = image_noise(1, 128, device = 0)

optimizer = optim.Adam([noise], lr=0.01)

imgs_over_time = []

iterations = 100
save_every = 1
for i in range(iterations):
    optimizer.zero_grad()
    # Having the default 0.75 trunc_psi causes gradients to be lost, so we must use None
    styles  = noise_to_styles(model, noise, trunc_psi = None)  # pass through mapping network
    images  = styles_to_images(model, styles, img_noises) # call the generator on intermediate style vectors
    images = images[:,0:1,:,:]
        
    if(i % save_every == 0):
        imgs_over_time.append(get_img_from_tensor(images, nrow=3))
    
    losses = [
        horizontal_orientation_loss(images),
        -15*smoothness_loss(images)
    ]
    
    k = 0
    for loss in losses:            
        loss.backward(retain_graph=True)
        print("%i/%i: loss%i=%0.04f" % (i+1, iterations, k, loss.item()))
        k += 1
    

    optimizer.step()



imageio.mimwrite("images_over_time.gif", imgs_over_time)