import pydicom 
import matplotlib.pyplot as plt 
import os 
import torch 
import numpy as np 

from losses import generalized_distance_transform

img = torch.ones((1, 100, 100, 1))
img[:, 10:20, 10:20, :] = 0.
img = img.cuda()
output = generalized_distance_transform(img)

output = output.cpu().numpy().reshape((100, 100))
plt.figure()
plt.imshow(output)
plt.show()
