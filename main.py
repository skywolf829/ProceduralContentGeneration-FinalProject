import torch
import numpy as np
import os
import matplotlib.pyplot as plt

# data from http://www.viewfinderpanoramas.org/dem3.html
# http://viewfinderpanoramas.org/Coverage%20map%20viewfinderpanoramas_org3.htm
byteorder = "big"

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
