import glob
import mediapy as media
from natsort import natsorted
from PIL import Image, ImageDraw, ImageFont


videos = {}
estim_path = []
paths = glob.glob("logs/cutting_tissues_twice/nerf_onlytest_time_127000/gt/*.rgb.png")
paths = natsorted(paths)
paths1 = paths[:40]
paths2 = paths[53:105]
paths3 = paths[105:156]
rot1 = glob.glob("logs/cutting_tissues_twice/nerf_onlytest_time_127000/novel/pose_52_*.rgb.png")
rot1 = natsorted(rot1)

rot2 = glob.glob("logs/cutting_tissues_twice/nerf_onlytest_time_127000/novel/pose_104_*.rgb.png")
rot2 = natsorted(rot1)
# print(paths)
images = []
for path in paths1:
  image = media.read_image(path)[:,:,:3]
  images.append(image)
  estim_path.append(path)

image = Image.open(path)
draw = ImageDraw.Draw(image)

symbol_size = 40
symbol_position = (10, 10)

# Draw the pause symbol (two vertical bars)
symbol_width = 10
symbol_height = 30
symbol_spacing = 10
symbol_color = (255, 0, 0)  # Red color (you can change the color)

# Draw the left bar
left_bar = (symbol_position[0], symbol_position[1], symbol_position[0] + symbol_width, symbol_position[1] + symbol_height)
draw.rectangle(left_bar, fill=symbol_color)

# Draw the right bar
right_bar = (symbol_position[0] + symbol_width + symbol_spacing, symbol_position[1], symbol_position[0] + 2 * symbol_width + symbol_spacing, symbol_position[1] + symbol_height)
draw.rectangle(right_bar, fill=symbol_color)

output_path = 'logs/cutting_tissues_twice/nerf_onlytest_time_127000/paused/pause.jpg'
image.save(output_path)
image = media.read_image(output_path)[:,:,:3]
images.append(image)
images.append(image)
images.append(image)
images.append(image)
images.append(image)
images.append(image)
images.append(image)
images.append(image)
images.append(image)
images.append(image)
# for path in rot1:
#   image = media.read_image(path)[:,:,:3]
#   images.append(image)
#   images.append(image)
#   images.append(image)
#   estim_path.append(path)
# for path in paths2:
#   image = media.read_image(path)[:,:,:3]
#   images.append(image)
#   estim_path.append(path)
# for path in rot2:
#   image = media.read_image(path)[:,:,:3]
#   images.append(image)
#   estim_path.append(path)
# for path in paths3:
#   image = media.read_image(path)[:,:,:3]
#   images.append(image)
#   estim_path.append(path)
estim_images = images

video = images
print(len(images))
media.write_video('logs/cutting_tissues_twice/nerf_onlytest_time_127000/gt.mp4', video, fps=7, qp=18)