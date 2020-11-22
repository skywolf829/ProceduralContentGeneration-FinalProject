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
import time
import zmq

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
    num_layers = model.GAN.G.num_layers
    w_def = [(w, num_layers)]

    w_tensors = styles_def_to_tensor(w_def)
    print(noise.shape)
    images = model.GAN.G(w_tensors, noise)
    images.clamp_(0., 1.)
    return images

def get_img_from_tensor(t, nrow=8):
    t = torch.clamp(t,0.0,1.0)
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

def smoothness_loss(imgs, mask=None):
    if(mask is None):
        mask = torch.ones(imgs.shape, device="cuda:0")
    laps = laplacian(imgs,"cuda:0")
    laps = torch.abs(laps)
    return (laps*mask).mean()

def ridgidness_loss(imgs, mask=None):
    if(mask is None):
        mask = torch.ones(imgs.shape, device="cuda:0")
    laps = laplacian(imgs,"cuda:0")
    laps = torch.abs(laps)
    return (-laps*mask).mean()

def exact_height_loss(imgs, mask=None):
    if(mask is None):
        mask = torch.ones(imgs.shape, device="cuda:0")
    return ((imgs-mask)**2).mean()
    
def lower_heightmap_loss(imgs, mask=None):
    if(mask is None):
        mask = torch.ones(imgs.shape, device="cuda:0")
    return (imgs*mask).mean()

def higher_heightmap_loss(imgs, mask=None):
    if(mask is None):
        mask = torch.ones(imgs.shape, device="cuda:0")
    return (-imgs*mask).mean()

def increase_vertical_orientation_loss(imgs, mask=None):
    if(mask is None):
        mask = torch.ones(imgs.shape, device="cuda:0")
    sobel_y = torch.abs(sobel(imgs, 0, "cuda:0"))
    return (-sobel_y*mask).mean()

def decrease_vertical_orientation_loss(imgs, mask=None):
    if(mask is None):
        mask = torch.ones(imgs.shape, device="cuda:0")
    sobel_y = torch.abs(sobel(imgs, 0, "cuda:0"))
    return (sobel_y*mask).mean()

def increase_horizontal_orientation_loss(imgs, mask=None):
    if(mask is None):
        mask = torch.ones(imgs.shape, device="cuda:0")
    sobel_x = torch.abs(sobel(imgs, 1, "cuda:0"))
    return (-sobel_x*mask).mean()

def decrease_horizontal_orientation_loss(imgs, mask=None):
    if(mask is None):
        mask = torch.ones(imgs.shape, device="cuda:0")
    sobel_x = torch.abs(sobel(imgs, 1, "cuda:0"))
    return (sobel_x*mask).mean()

def load_latest_model():
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
    model = Trainer(**model_args)
    model.load(-1)
    model.GAN.train(False)
    return model

def optimize_on(model, noise, img_noises, losses, masks, iterations, save_every = 1):

    optimizer = optim.Adam([noise], lr=0.01)
    imgs_over_time = []

    for i in range(iterations):
        optimizer.zero_grad()
        # Having the default 0.75 trunc_psi causes gradients to be lost, so we must use None
        styles  = noise_to_styles(model, noise, trunc_psi = None)  # pass through mapping network
        images  = styles_to_images(model, styles, img_noises) # call the generator on intermediate style vectors
        images = images[:,0:1,:,:]
            
        if(i % save_every == 0):
            imgs_over_time.append(get_img_from_tensor(images, nrow=8))
        
        _losses = []
        for j in range(len(losses)):
            _losses.append(losses[j](images, masks[j]))
        
        j = 0
        for loss in _losses:            
            loss.backward(retain_graph=True)
            print("%i/%i: loss%i=%0.04f" % (i+1, iterations, j, loss.item()))
            j += 1
        
        optimizer.step()
    return imgs_over_time, noise

def optimize_on_multistyle(model, noises, freeze_noises, img_noises, 
losses, masks, iterations, save_every = 1):
    optimize_on = []
    for i in range(len(noises)):
        if(not freeze_noises[i]):
            optimize_on.append(noises[i])
    optimizer = optim.Adam(optimize_on, lr=0.01)
    imgs_over_time = []

    for i in range(iterations):
        optimizer.zero_grad()
        # Having the default 0.75 trunc_psi causes gradients to be lost, so we must use None
        multistyles  = noises_to_multistyles(model, noises, trunc_psi = None)  # pass through mapping network
        images  = multistyles_to_images(model, multistyles, img_noises) # call the generator on intermediate style vectors
        images = images[:,0:1,:,:]
            
        if(i % save_every == 0):
            imgs_over_time.append(get_img_from_tensor(images, nrow=8))
        
        _losses = []
        for j in range(len(losses)):
            _losses.append(losses[j](images, masks[j]))
        
        j = 0
        for loss in _losses:            
            loss.backward(retain_graph=True)
            print("%i/%i: loss%i=%0.04f" % (i+1, iterations, j, loss.item()))
            j += 1
        
        optimizer.step()
    return imgs_over_time, noises

def optimize_with(model, optimizer, noise, img_noises, 
losses, masks, iterations, heightmap_full_resolution, save_every=1):
    if(optimizer is None):
        optimizer = optim.Adam([noise], lr=0.01)
    imgs_over_time = []

    for i in range(iterations):
        optimizer.zero_grad()
        # Having the default 0.75 trunc_psi causes gradients to be lost, so we must use None
        styles  = noise_to_styles(model, noise, trunc_psi = None)  # pass through mapping network
        images  = styles_to_images(model, styles, img_noises) # call the generator on intermediate style vectors
        images = images[:,0:1,:,:]
        img_resized = F.interpolate(images, 
        size=[heightmap_full_resolution, heightmap_full_resolution],
        mode='bicubic', align_corners=False)

        if(i % save_every == 0):
            imgs_over_time.append(get_img_from_tensor(img_resized, nrow=8))
        
        _losses = []
        for j in range(len(losses)):
            _losses.append(losses[j](images, masks[j]))
        
        j = 0
        for loss in _losses:            
            loss.backward(retain_graph=True)
            print("%i/%i: loss%i=%0.04f" % (i+1, iterations, j, loss.item()))
            j += 1
        
        optimizer.step()
    return imgs_over_time, noise, optimizer

def optimize_with_multistyle(model, optimizer, noises, freeze_noises, img_noises, 
losses, masks, iterations, heightmap_full_resolution, save_every=1):
    optimize_on = []
    
    for i in range(len(noises)):
        if(not freeze_noises[i]):
            optimize_on.append(noises[i])

    if(optimizer is None):
        optimizer = optim.Adam(optimize_on, lr=0.01)
    imgs_over_time = []

    for i in range(iterations):
        optimizer.zero_grad()
        # Having the default 0.75 trunc_psi causes gradients to be lost, so we must use None
        multistyles  = noises_to_multistyles(model, noises, trunc_psi = None)  # pass through mapping network
        images  = multistyles_to_images(model, multistyles, img_noises) # call the generator on intermediate style vectors
        images = images[:,0:1,:,:]
        img_resized = F.interpolate(images, 
        size=[heightmap_full_resolution, heightmap_full_resolution],
        mode='bicubic', align_corners=False)

        if(i % save_every == 0):
            imgs_over_time.append(get_img_from_tensor(img_resized, nrow=8))
        
        _losses = []
        for j in range(len(losses)):
            _losses.append(losses[j](images, masks[j]))
        
        j = 0
        for loss in _losses:            
            loss.backward(retain_graph=True)
            print("%i/%i: loss%i=%0.04f" % (i+1, iterations, j, loss.item()))
            j += 1
        
        optimizer.step()
    return imgs_over_time, noises, optimizer

def feed_forward(model, noise, img_noises, heightmap_full_resolution):
    styles  = noise_to_styles(model, noise, trunc_psi = None)
    images  = styles_to_images(model, styles, img_noises) 
    images = images[:,0:1,:,:]
    img_resized = F.interpolate(images, 
    size=[heightmap_full_resolution, heightmap_full_resolution],
    mode='bicubic', align_corners=False)
    return get_img_from_tensor(img_resized, nrow=8)

def noises_to_multistyles(model, noises, trunc_psi=0.75):
    ws = []
    for i in range(len(noises)):
        w = model.GAN.S(noises[i])
        if exists(trunc_psi):
            print("truncing")
            w = model.truncate_style(w)
        ws.append(w)
    return ws

def multistyles_to_images(model, ws, noise):
    num_layers = model.GAN.G.num_layers
    final_w = []
    for i in range(len(ws)):
        w_tensors = styles_def_to_tensor([(ws[i], num_layers)])
        final_w.append(w_tensors[:,i:i+1,:])
    w_tensors = torch.cat(final_w, dim=1)
    images = model.GAN.G(w_tensors, noise)
    images.clamp_(0., 1.)
    return images

def feed_forward_multistyle(model, noises, img_noises, heightmap_full_resolution):
    multistyles  = noises_to_multistyles(model, noises, trunc_psi = None)
    images  = multistyles_to_images(model, multistyles, img_noises) 
    images = images[:,0:1,:,:]
    img_resized = F.interpolate(images, 
    size=[heightmap_full_resolution, heightmap_full_resolution],
    mode='bicubic', align_corners=False)
    return get_img_from_tensor(img_resized, nrow=8)

if __name__ == '__main__':
    model = load_latest_model()

    num_image_tiles = 8
    results_dir = './results'
    name = 'default'
    
    iterations = 100
    img_resolution = 128
    num_images = 64

    noise   = torch.randn(num_images, 512).cuda()    
    img_noises = image_noise(1, 128, device = 0)
    noise.requires_grad = True
    #losses = [lower_heightmap_loss]
    #masks = [torch.ones([num_images, 1, img_resolution, img_resolution])]
    #imgs_over_time, noise = optimize_on(model, noise, img_noises, losses, masks, iterations)
    imgs = feed_forward(model, noise, img_noises, 128)
    imageio.imwrite("random_imgs.png", imgs)
