import numpy as np
from natsort import natsorted
import shutil
import os
import glob
videos = {}
paths = glob.glob("data/Hamlyn/image_depth_data/rectified04_1/masks/*.jpg")
paths = natsorted(paths)
print(paths)
for i,f in enumerate(paths):
    tmp_str = '0000000000'
    num = str(i)
    print(f)
    print("----------------")
    src = f
    dct = os.path.join("data/Hamlyn/image_depth_data/rectified04_1/images_formatteed/", tmp_str[:10-len(num)]+num+".jpg")
    print(src,dct)
    shutil.move(src, dct)
