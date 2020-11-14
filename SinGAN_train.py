import torch
from torch import nn, optim
import torch.functional as F
import os
import numpy as np
from utility import *
from SinGAN_model import *
import argparse
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--epochs',type=int,default=1,help="Number of epochs to train")
    parser.add_argument('--batch',type=int,default=1,help="Batch size")
    parser.add_argument('--num_workers',type=int,default=16,help="Number of workers to load data in parallel")
    parser.add_argument('--training_folder',type=str,default="HeightMapImages",help="Folder that has training data")
    parser.add_argument('--device',type=str,default="cuda:0",help="What device to use for training")
    parser.add_argument('--load_from',type=str,default=None,help="If resuming training, where to load from")
    parser.add_argument('--log_every',type=int,default=1,help="How often to log using tensorboard")
    parser.add_argument('--save_every',type=int,default=1,help="How often (# epochs) to save the model")
    parser.add_argument('--save_name',type=str,default="Temp",help="Name to save the model as")
    args = vars(parser.parse_args())

    dataset = Dataset(args['training_folder'])
    dataloader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=args['batch'],
                    shuffle=True,
                    num_workers=args['num_workers']
                )
    if(args['load_from'] is not None):
        g, d = load_models(os.path.join("SavedModels", 
        args['load_from']), args['device'])
        g = g.to(args['device'])
        d = d.to(args['device'])
    else:
        n, gs, ds = init_scales(dataset, args['device'])

    for s in range(n):
        print("Starting scale %i with resolution %s" % (s, str(gs[s].resolution)))
        g = gs[s]
        d = ds[s]
        generator_optimizer = optim.Adam(g.parameters(), 
        lr=0.0005, betas=(0.5,0.99))
        discriminator_optimizer = optim.Adam(d.parameters(), 
        lr=0.0005, betas=(0.5,0.99))

        writer = SummaryWriter(os.path.join('tensorboard',
        args['save_name']+"_scale"+str(s)))

        iteration = 0
        for epoch in range(args['epochs']):
            print("Starting epoch %i" % (epoch))
            for i, (real_heightmaps) in enumerate(dataloader):
                real_heightmaps = real_heightmaps.to(args['device'])
                real_heightmaps = laplace_pyramid_downscale2D(real_heightmaps, n-s-1,
                0.75, args['device'])
                if(s > 0):
                    fake_heightmaps = generate(gs[0:s], 
                    real_heightmaps.shape[0], args['device'])
                    fake_heightmaps= F.interpolate(fake_heightmaps, 
                    size=g.resolution, mode="bilinear", 
                    align_corners=False)
                else:
                    fake_heightmaps = torch.zeros(real_heightmaps.shape, 
                    device=args['device'])

                rand_input = torch.randn([real_heightmaps.shape[0], 1,
                g.resolution[0], g.resolution[1]], 
                device=args['device'])
                fake_heightmaps = g(fake_heightmaps, rand_input)

                # Update discriminator: maximize D(x) + D(G(z))            
                for j in range(3):            
                    d.zero_grad()

                    # Train with real 
                    output = d(real_heightmaps)
                    discrim_error_real = -output.mean()
                    discrim_error_real.backward(retain_graph=True)
                    
                    # Train with the generated image
                    output = d(fake_heightmaps.detach())
                    discrim_error_fake = output.mean()
                    discrim_error_fake.backward(retain_graph=True)
                    
                    # Regularization required for WGAN
                    gradient_penalty = calc_gradient_penalty(d, 
                    real_heightmaps, fake_heightmaps, args['device'])
                    gradient_penalty.backward(retain_graph=True)

                    discriminator_optimizer.step()
                
                for j in range(3):
                    g.zero_grad()

                    # Train to trick the discriminator
                    if(s > 0):
                        fake_heightmaps = generate(gs[0:s], 
                        real_heightmaps.shape[0], args['device'])
                        fake_heightmaps= F.interpolate(fake_heightmaps, 
                        size=g.resolution, mode="bilinear", 
                        align_corners=False)
                    else:
                        fake_heightmaps = torch.zeros(real_heightmaps.shape, 
                        device=args['device'])

                    rand_input = torch.randn([real_heightmaps.shape[0], 1,
                    g.resolution[0], g.resolution[1]], 
                    device=args['device'])
                    fake_heightmaps = g(fake_heightmaps, rand_input)
                    output = d(fake_heightmaps)
                    generator_error = -output.mean()
                    generator_error.backward(retain_graph=True)
                                
                    # Train to be conditioned by input values
                    # To be implemented

                    generator_optimizer.step()
                    
                if(iteration % args['log_every'] == 0):
                    print("%i: D_loss: %0.04f, G_loss: %0.04f" % \
                    (iteration, discrim_error_fake.item() + discrim_error_real.item(), 
                    generator_error.item()))

                    writer.add_scalar('D_loss',
                    discrim_error_real.item() + discrim_error_fake.item(), 
                    iteration) 
                    writer.add_scalar('G_loss',
                    generator_error.item(), 
                    iteration) 

                    writer.add_image('fake', 
                        (fake_heightmaps[0] - fake_heightmaps[0].min()) \
                        / (fake_heightmaps[0].max() - fake_heightmaps[0].min()), 
                        iteration)
                    writer.add_image('real', 
                        (real_heightmaps[0] - real_heightmaps[0].min()) \
                        / (real_heightmaps[0].max() - real_heightmaps[0].min()), 
                        iteration)       

                iteration += 1

            if(epoch % args['save_every'] == 0):
                save_models(gs, ds, args['save_name'])
        
        ds[s] = ds[s].to("cpu")
        gs[s] = gs[s].eval()
        ds[s] = ds[s].eval()