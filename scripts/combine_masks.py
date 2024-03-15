import cv2
from os import listdir
from os.path import isfile, join

mypath = "data/Hamlyn/image_depth_data/rectified05_1/masks_bad"
joinpath = "data/Hamlyn/image_depth_data/rectified05_1/masks"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for f in onlyfiles:
    ref1 = cv2.imread(join(mypath, f))
    ref2 = cv2.imread("data/Hamlyn/image_depth_data/rectified05_1/refer1.jpg")
    # print(ref1,ref2)
    and_img = cv2.bitwise_not(cv2.bitwise_and(cv2.bitwise_not(ref1),cv2.bitwise_not(ref2)))
    cv2.imwrite(join(joinpath, f), and_img)