from PIL import Image

img = Image.open("data/Hamlyn/image_depth_data/rectified05_1/rectified05.jpg") # get image
pixels = img.load() # create the pixel map

for i in range(img.size[0]): # for every pixel:
    for j in range(img.size[1]):
        if i<10:
            pixels[i,j] = (0,0,0)
            continue
        if pixels[i,j][0]>250 and pixels[i,j][1]>250 and pixels[i,j][2]>250: # if not black:
            pixels[i,j] = (255, 255, 255) # change to white

for i in range(img.size[0]): # for every pixel:
    for j in range(img.size[1]):
        if pixels[i,j] == (0,0,0): # if not black:
            pixels[i,j] = (255, 255, 255) 
        else:
            pixels[i,j] = (0,0,0)

img.save("data/Hamlyn/image_depth_data/rectified05_1/refer1.jpg")