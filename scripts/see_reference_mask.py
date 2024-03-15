from PIL import Image
from os import listdir
from os.path import isfile, join


img = Image.open("data/Hamlyn/image_depth_data/rectified04_1/masks/refer.jpg") # get image
pixels = img.load() 


mypath = "data/Hamlyn/image_depth_data/rectified04_1/images"
refpath = "data/Hamlyn/image_depth_data/rectified04_1/references"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for f in onlyfiles:
    img1 = Image.open(join(mypath, f))
    pixels1 = img1.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if pixels[i,j] == (0,0,0):
                pixels1[i,j] = (0,0,0)
    img1.save(join(refpath, f))