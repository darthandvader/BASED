import os
from os import listdir
from os.path import isfile, join
import shutil
from natsort import natsorted

data_path = 'data/Hamlyn/image_depth_data/rectified17/image01/'
dst = 'data/Hamlyn/image_depth_data/rectified17/images/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
onlyfiles = natsorted(onlyfiles)



for i,file in enumerate(onlyfiles[2600:]):
    src = data_path + file 
    if i%5 == 0 and i<=605:
        dst1 = dst + file
    else:
        continue
    shutil.copy2(src, dst1)

