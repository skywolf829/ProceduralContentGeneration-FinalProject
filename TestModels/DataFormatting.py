import torch
import numpy as np
import os
import matplotlib.pyplot as plt

'''
# data from http://www.viewfinderpanoramas.org/dem3.html
# http://viewfinderpanoramas.org/Coverage%20map%20viewfinderpanoramas_org3.htm
byteorder = "big" # Doesn't matter, it looks at a single byte at a time

for foldername in os.listdir("./RawData"):
    for filename in os.listdir(os.path.join("./RawData", foldername)):
        print(filename.split(".")[0])
        a = np.fromfile(os.path.join('./RawData', foldername, filename), dtype=np.byte)

        res = int((a.shape[0]/2)**0.5)
        b = np.zeros([res, res])

        for i in range(int(a.shape[0]/2)):
            row = int(i / res)
            col = i % res
            b[row, col] = 256*int.from_bytes(a[i*2], byteorder) + int.from_bytes(a[(i*2)+1], byteorder)
        
        np.save(os.path.join("./NumpyRawData", foldername + "_" + filename.split(".")[0]+".npy"), b)


'''

'''
for filename in os.listdir("./NumpyRawData"):
    a = np.load(os.path.join("./NumpyRawData", filename))
    for x in range(0, a.shape[0], 512):
        x_start = x
        x_end = x_start+512
        if(x+512 > a.shape[0]):
            x_start = a.shape[0]-512
            x_end = a.shape[0]

        for y in range(0, a.shape[1], 512):
            y_start = y
            y_end = y_start+512
            if(y+512 > a.shape[1]):
                y_start = a.shape[1]-512
                y_end = a.shape[1]

            np.save(os.path.join("./NumpyRawDataCropped", filename \
            + "_" + str(x) + "_" + str(y)+".npy"), \
            a[x_start:x_end, y_start:y_end])


'''
import imageio
for filename in os.listdir("./NumpyRawDataCropped"):
    print(filename)
    a = np.load(os.path.join("./NumpyRawDataCropped", filename))
    a -= a.min()
    a /= a.max()
    a *= 255
    a = a.astype(np.uint8)
    n = filename.split(".npy")[0] + filename.split(".npy")[1]
    n = os.path.join("./HeightMapImages", n +".png")
    imageio.imwrite(n, a)

        