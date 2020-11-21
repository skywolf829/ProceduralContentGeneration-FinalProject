from sampling import *
import time
import zmq
import base64
import imageio

def optimize_one_step(model, optimizer, noise, 
img_noises, losses, masks, heightmap_full_resolution):
    imgs_over_time, noise, optimizer = \
    optimize_with(model, optimizer, noise, img_noises, 
    losses, masks, 1, heightmap_full_resolution)

    img_to_send = imgs_over_time[-1].copy()
    img_to_send = bytearray(img_to_send)
    img_to_send = base64.b64encode(img_to_send)
    return noise, img_to_send, imgs_over_time[-1], optimizer

def request_and_masks_to_losses_and_masks(request_list):
    losses = []
    masks = []
    if(len(request_list) == 0 or request_list[0] == ''):
        print("Error with request list, no elements")
        losses.append(lower_heightmap_loss)
        masks.append(torch.ones([num_images, 1, img_resolution, img_resolution], 
        device="cuda:0"))
    else:
        spot = 0
        while(spot < len(request_list)):
            mask = torch.zeros([num_images, 1, img_resolution, img_resolution], 
            device="cuda:0")
            if(request_list[spot] == "Raise"):
                losses.append(higher_heightmap_loss)
            elif(request_list[spot] == "Lower"):
                losses.append(lower_heightmap_loss)
            elif(request_list[spot] == "Exact height"):
                losses.append(exact_height_loss)
            elif(request_list[spot] == "Decrease vertical alignment"):
                losses.append(decrease_vertical_orientation_loss)
            elif(request_list[spot] == "Increase vertical alignment"):
                losses.append(increase_vertical_orientation_loss)
            elif(request_list[spot] == "Decrease horizontal alignment"):
                losses.append(decrease_horizontal_orientation_loss)
            elif(request_list[spot] == "Increase horizontal alignment"):
                losses.append(increase_horizontal_orientation_loss)
            elif(request_list[spot] == "Smooth"):
                losses.append(smoothness_loss)
            elif(request_list[spot] == "Ridged" or \
            request_list[spot] == "Ridges"):
                losses.append(ridgidness_loss)
            else:
                print("Loss " + str(request_list[spot]) + " not recognized.")
            spot += 1
            m = torch.tensor(np.array(request_list[ \
            spot:spot+img_resolution*img_resolution], dtype=np.float32)).to("cuda:0") / 255.0
            m = m.reshape([img_resolution, img_resolution])
            mask[:,:,:,:] = m
            masks.append(mask)
            spot += img_resolution*img_resolution
    return losses, masks

def request_to_losses_and_masks(request_list):
    losses = []
    masks = []
    if(len(request_list) == 0 or request_list[0] == ''):
        print("Error with request list, no elements")
        losses.append(lower_heightmap_loss)
        masks.append(torch.ones([num_images, 1, img_resolution, img_resolution], 
        device="cuda:0"))
    else:
        spot = 0
        while(spot < len(request_list)):
            mask = torch.zeros([num_images, 1, img_resolution, img_resolution], 
            device="cuda:0")
            if(request_list[spot] == "Raise"):
                losses.append(higher_heightmap_loss)
            elif(request_list[spot] == "Lower"):
                losses.append(lower_heightmap_loss)
            elif(request_list[spot] == "Decrease vertical alignment"):
                losses.append(decrease_vertical_orientation_loss)
            elif(request_list[spot] == "Increase vertical alignment"):
                losses.append(increase_vertical_orientation_loss)
            elif(request_list[spot] == "Decrease horizontal alignment"):
                losses.append(decrease_horizontal_orientation_loss)
            elif(request_list[spot] == "Increase horizontal alignment"):
                losses.append(increase_horizontal_orientation_loss)
            elif(request_list[spot] == "Smooth"):
                losses.append(smoothness_loss)
            elif(request_list[spot] == "Ridged"):
                losses.append(ridgidness_loss)
            else:
                print("Loss " + str(request_list[spot]) + " not recognized.")
            
            mask[:,:,int(request_list[spot+3]):int(request_list[spot+4]),
            int(request_list[spot+1]):int(request_list[spot+2])] = 1
            masks.append(mask.clone())
            spot += 5
    return losses, masks

if __name__ == '__main__':
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    model = load_latest_model()

    num_image_tiles = 8
    results_dir = './results'
    name = 'default'
    
    iterations = 100
    num_images = 1
    img_resolution = 128
    heightmap_full_resolution = 128

    training_heightmap = False
    curr_iter = 0

    noise   = torch.randn(num_images, 512).cuda()
    img_noises = image_noise(1, 128, device = 0)
    noise.requires_grad = True

    last_noise = noise.clone()
    last_img_noises = img_noises.clone()

    try:
        while True:
            #  Wait for next request from client
            message = socket.recv()
            message = message.decode('ascii').split(",")
            print("Received " + str(message[0]))
            if(message[0] == "requesting_heightmap"):                
                if(not training_heightmap):
                    training_heightmap = True
                    curr_iter = 0
                    img_seq = []
                    losses, masks = request_to_losses_and_masks(message[1:])
                    last_noise = noise.clone()
                    last_img_noises = img_noises.clone()
                    noise, img_to_send, img_to_save, optimizer = \
                    optimize_one_step(model, None, noise, img_noises, 
                    losses, masks, heightmap_full_resolution)
                    socket.send(img_to_send)
                    img_seq.append(img_to_save)
                    curr_iter += 1
                    print("Sent image")
                elif(curr_iter < iterations):
                    noise, img_to_send, img_to_save, optimizer = \
                    optimize_one_step(model, optimizer, noise, 
                    img_noises, losses, masks, heightmap_full_resolution)
                    socket.send(img_to_send)
                    img_seq.append(img_to_save)
                    curr_iter += 1
                    print("Sent image")
                else:
                    training_heightmap = False
                    imageio.imwrite("sentimage.png", img_seq[-1])
                    imageio.mimwrite("imgseq.gif", img_seq)
                    socket.send_string("finished")
            if(message[0] == "requesting_heightmap_with_masks"):
                if(not training_heightmap):
                    training_heightmap = True
                    curr_iter = 0
                    img_seq = []
                    losses, masks = request_and_masks_to_losses_and_masks(message[1:])
                    last_noise = noise.clone()
                    last_img_noises = img_noises.clone()
                    noise, img_to_send, img_to_save, optimizer = \
                    optimize_one_step(model, None, noise, img_noises, 
                    losses, masks, heightmap_full_resolution)
                    socket.send(img_to_send)
                    img_seq.append(img_to_save)
                    curr_iter += 1
                    print("Sent image")
                elif(curr_iter < iterations):
                    noise, img_to_send, img_to_save, optimizer = \
                    optimize_one_step(model, optimizer, noise, 
                    img_noises, losses, masks, heightmap_full_resolution)
                    socket.send(img_to_send)
                    img_seq.append(img_to_save)
                    curr_iter += 1
                    print("Sent image")
                else:
                    training_heightmap = False
                    imageio.imwrite("sentimage.png", img_seq[-1])
                    imageio.mimwrite("imgseq.gif", img_seq)
                    socket.send_string("finished")
            if(message[0] == "new_noise"):
                last_noise = noise.clone()
                last_img_noises = img_noises.clone()
                noise   = torch.randn(num_images, 512).cuda()
                img_noises = image_noise(1, 128, device = 0)
                noise.requires_grad = True
                img = feed_forward(model, noise, 
                img_noises, heightmap_full_resolution)
                img_to_send = img.copy()
                img_to_send = bytearray(img_to_send)
                img_to_send = base64.b64encode(img_to_send)
                imageio.imwrite("sentimage.png", img)
                socket.send(img_to_send)
            if(message[0] == "undo"):
                last_noise_copy = last_noise.detach().clone()
                last_noise = noise
                noise = last_noise_copy
                noise.requires_grad = True

                last_img_noises_copy = last_img_noises.detach().clone()
                last_img_noises = img_noises
                img_noises = last_img_noises_copy
                
                img = feed_forward(model, noise, 
                img_noises, heightmap_full_resolution)
                img_to_send = img.copy()
                img_to_send = bytearray(img_to_send)
                img_to_send = base64.b64encode(img_to_send)
                imageio.imwrite("sentimage.png", img)
                socket.send(img_to_send)
            if(message[0] == "stop"):
                print("Stopping")
                socket.send_string("stopping")
                quit()    
            if(message[0] == "update_resolution"):
                heightmap_full_resolution = int(message[1])
                img = feed_forward(model, noise, 
                img_noises, heightmap_full_resolution)
                img_to_send = img.copy()
                img_to_send = bytearray(img_to_send)
                img_to_send = base64.b64encode(img_to_send)
                imageio.imwrite("sentimage.png", img)
                socket.send(img_to_send)

            
    except KeyboardInterrupt:
        print("Ctrl-C - terminated")
