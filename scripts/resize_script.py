from PIL import Image
import os
from os import listdir
from os.path import isfile, join
from natsort import natsorted

image = Image.open('data/InHouse/chicken_tissue_1/images/left_0.png')
data_path = 'data/InHouse/chicken_tissue_1/images/'
data_path_new = 'data/InHouse/chicken_tissue_2/images/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
onlyfiles = natsorted(onlyfiles)
i = image.size   # current size (height,width)
i = i[0]//3, i[1]//3  # new size
for file in onlyfiles:
    print(data_path+file)
    newimage = Image.open(data_path+file)
    newimage = newimage.resize(i)
    newimage.save(data_path_new+file)