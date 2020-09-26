import random
from glob import glob
from os.path import *

import torch
import torchvision
import numpy as np
from PIL import Image

from skimage.io import imread
from skimage import color
from skimage.transform import resize
from sklearn.neighbors import NearestNeighbors

# Verify if a GPU is available.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Read randomly images from the file path.
root = "/home/mfcs/mestrado_projeto/pytorch_image_colorization_mfcs/dataset/train/images/"
filename_lists = sorted(glob(join(root, "*.jpg")))
random.shuffle(filename_lists)

# Load the 313 bins of color.
# points shape -> (313, 2).
# Directory of pts_in_hull.npy file
points = np.load("/home/mfcs/mestrado_projeto/pytorch_image_colorization_mfcs/code/resources/pts_in_hull.npy")
points = points.astype(np.float64)
points = torch.from_numpy(points)

# Create a vector containing 313 elements(zeros) in which each zero will be replaced by the bin probability in the dataset.
probs = np.zeros((points.shape[0]), dtype=np.float64)

# In this loop the tuples (a,b) are generated from the image and the distances of each tuple from the image
# to the bins of color are calculated and the bin count in the dataset is performed.
for index, img_file in enumerate(filename_lists):
    print(index, img_file)
    img_rgb = Image.open(img_file)
    img_rgb = np.asarray(img_rgb)
    img_lab = color.rgb2lab(img_rgb)
    img_lab = img_lab.transpose(2,0,1)
    #img_lab = torch.from_numpy(img_lab)
    img_l = img_lab[0,:,:]          # L channel
    img_ab = img_lab[1:3,:,:]       # (a,b) channel
    img_a = img_ab[0,:,:]           # a channel
    img_b = img_ab[1:2,:,:]         # b channel
    img_a_flatten = img_a.flatten()
    img_b_flatten = img_b.flatten()
    list_of_ab_colors = [[img_a_flatten[i], img_b_flatten[i]] for i in range(0, len(img_a_flatten))]

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points)
    (dists, inds) = nbrs.kneighbors(list_of_ab_colors)

    for i in inds:
        j = int(i[0])
        probs[j] += 1
        #print(j)

# The probabilites for each bin is performed.
probs = probs / np.sum(probs)
probs = probs + 0.000001
print(probs)

# Print all bins with probabilities > 0.0
for i in range(len(probs)):
    if probs[i] !=0:
        print(i, probs[i])

# Save the final file .npy containing the probability of each bin of color from the dataset in use.
# Directory to save the .npy file
np.save("/home/mfcs/mestrado_projeto/pytorch_image_colorization_mfcs/code/resources/prior_probs", probs)