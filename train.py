import torch
from torch import nn, optim
import torch.functional as F
import os
import numpy as np
from utility import *
from model import *
import argparse
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--epochs',type=int,default=20,help="Number of epochs to train")
    parser.add_argument('--batch',type=int,default=1,help="Batch size")
    parser.add_argument('--num_workers',type=int,default=8,help="Number of workers to load data in parallel")
    parser.add_argument('--training_folder',type=str,default="NumpyRawDataCropped",help="Folder that has training data")
    parser.add_argument('--device',type=str,default="cuda:0",help="What device to use for training")
    parser.add_argument('--load_from',type=str,default=None,help="If resuming training, where to load from")
    parser.add_argument('--log_every',type=int,default=10,help="How often to log using tensorboard")
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
        g = generator(args['device']).to(args['device'])
        g.apply(weights_init)
        d = discriminator(args['device']).to(args['device'])
        d.apply(weights_init)


    generator_optimizer = optim.Adam(g.parameters(), 
    lr=0.0005, betas=(0.5,0.99))
    discriminator_optimizer = optim.Adam(d.parameters(), 
    lr=0.0005, betas=(0.5,0.99))

    writer = SummaryWriter(os.path.join('tensorboard',args['save_name']))

    iteration = 0
    for epoch in range(args['epochs']):
        print("Starting epoch %i" % (epoch))
        for i, (real_heightmaps) in enumerate(dataloader):
            real_heightmaps = real_heightmaps.to(args['device'])
            rand_input = torch.randn([real_heightmaps.shape[0], 16], 
            device=args['device'])
            generated = g(rand_input)
            # Update discriminator: maximize D(x) + D(G(z))            
            for j in range(3):            
                d.zero_grad()

                # Train with real 
                output = d(real_heightmaps)
                discrim_error_real = -output.mean()
                discrim_error_real.backward(retain_graph=True)
                
                # Train with the generated image
                output = d(generated.detach())
                discrim_error_fake = output
                discrim_error_fake.backward(retain_graph=True)
                
                # Regularization required for WGAN
                gradient_penalty = calc_gradient_penalty(d, 
                real_heightmaps, generated, args['device'])
                gradient_penalty.backward(retain_graph=True)

                discriminator_optimizer.step()
            
            for j in range(3):
                g.zero_grad()

                # Train to trick the discriminator
                rand_input = torch.randn([real_heightmaps.shape[0], 16], 
                device=args['device'])
                generated = g(rand_input)
                output = d(generated)
                generator_error = -output
                generator_error.backward(retain_graph=True)
                            
                # Train to be conditioned by input values
                # To be implemented

                generator_optimizer.step()
                
            if(iteration % args['log_every'] == 0):
                writer.add_scalar('D_loss',
                discrim_error_real.item() + discrim_error_fake.item(), 
                iteration) 
                writer.add_scalar('G_loss',
                generator_error.item(), 
                iteration) 

                writer.add_image('generated', 
                    (generated[0] - generated[0].min()) \
                     / (generated[0].max() - generated[0].min()), 
                     iteration)
                writer.add_image('real', 
                     (real_heightmaps[0] - real_heightmaps[0].min()) \
                     / (real_heightmaps[0].max() - real_heightmaps[0].min()), 
                     iteration)
                     

            iteration += 1

        if(epoch % args['save_every'] == 0):
            save_models(g, d, args['save_name'])