import cv2
from os import listdir
from os.path import isfile, join


m1 = cv2.imread("data/Hamlyn/image_depth_data/rectified18_1/frame0_refer1.jpg")
m2 = cv2.imread("data/Hamlyn/image_depth_data/rectified18_1/frame0_refer2.jpg")

joinpath = "data/Hamlyn/image_depth_data/rectified18_2/masks"
surprose = "data/Hamlyn/image_depth_data/rectified18_2/masks1"
onlyfiles = [f for f in listdir(joinpath) if isfile(join(joinpath, f))]
ref = "data/Hamlyn/image_depth_data/rectified18_2/ref"
for f in onlyfiles:
    ref1 = cv2.imread(join(joinpath, f))
    ref2 = cv2.imread("data/Hamlyn/image_depth_data/rectified18_2/images/"+f)
    print(ref1.shape, m1.shape)
    and_img = cv2.bitwise_and(cv2.bitwise_not(ref1),cv2.bitwise_not(m1))
    and_img = cv2.bitwise_and(and_img,cv2.bitwise_not(m2))
    for i in range(and_img.shape[0]):
        for j in range(and_img.shape[1]):
            if j<190 or j>550 or i>282:
                and_img[i][j] = (0,0,0)
    cv2.imwrite(join(surprose, f), cv2.bitwise_not(and_img))
    # for i in range(and_img.shape[0]):
    #     for j in range(and_img.shape[1]):
    #         if and_img[i][j][0] == 0 and and_img[i][j][1] == 0 and and_img[i][j][2] == 0:
    #             ref2[i][j] = (255,255,255)
    # cv2.imwrite(join(ref, f), ref2)