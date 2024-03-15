import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np 
from natsort import natsorted
from PIL import Image





onlyfiles = [f for f in listdir("data/Hamlyn/image_depth_data/rectified09/endodepth/") if isfile(join("data/Hamlyn/image_depth_data/rectified09/endodepth/", f))]
onlyfiles = natsorted(onlyfiles)
for file in onlyfiles:
    im = Image.open('data/Hamlyn/image_depth_data/rectified09/endodepth/' + file).convert('L')
    im.save('data/Hamlyn/image_depth_data/rectified09/depth/' + file)