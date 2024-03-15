import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np 
from natsort import natsorted


image = cv2.imread('data/Hamlyn/image_depth_data/rectified17/images/0000002600.jpg')
h, w, c = image.shape
blank_image2 = 0 * np.ones(shape=(h, w, c), dtype=np.uint8)


onlyfiles = [f for f in listdir("data/Hamlyn/image_depth_data/rectified17/images/") if isfile(join("data/Hamlyn/image_depth_data/rectified17/images/", f))]
onlyfiles = natsorted(onlyfiles)
for file in onlyfiles:
    cv2.imwrite('data/Hamlyn/image_depth_data/rectified17/masks/' + file, blank_image2)