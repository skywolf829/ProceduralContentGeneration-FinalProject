from sampling import *
import time
import zmq
import base64

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

    while True:
        #  Wait for next request from client
        message = socket.recv()
        print("Received request: %s" % message)

        noise   = torch.randn(num_images, 512).cuda()
        noise.requires_grad = True
        losses = [elevation_loss]
        masks = [torch.ones([num_images, 1, img_resolution, img_resolution])]
        imgs_over_time = optimize_on(model, noise, losses, masks, iterations)

        #  Send reply back to client
        img_to_send = imgs_over_time[-1].copy()
        img_to_send = bytearray(img_to_send)
        img_to_send = base64.b64encode(img_to_send)
        socket.send(img_to_send)
        print("Sent image")
