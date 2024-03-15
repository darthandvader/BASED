import glob
import mediapy as media
from natsort import natsorted
from PIL import Image, ImageDraw, ImageFont


videos = {}
estim_path = []
paths = glob.glob("logs/cutting_tissues_twice/nerf_onlytest_time_127000/estim/*.rgb.png")
paths = natsorted(paths)
paths1 = paths[40:53]
paths2 = paths[53:105]
paths3 = paths[105:156]
rot1 = glob.glob("logs/cutting_tissues_twice/nerf_onlytest_time_127000/novel/pose_52_*.rgb.png")
rot1 = natsorted(rot1)

rot2 = glob.glob("logs/cutting_tissues_twice/nerf_onlytest_time_127000/novel/pose_104_*.rgb.png")
rot2 = natsorted(rot1)
# print(paths)
images = []
for path in paths1:
  image = Image.open(path)
  draw = ImageDraw.Draw(image)
  symbol_size = 40
  symbol_position = (10, 10)
  triangle_coordinates = [
    (symbol_position[0], symbol_position[1]),  # Top vertex
    (symbol_position[0] + symbol_size, symbol_position[1] + symbol_size // 2),  # Bottom-right
    (symbol_position[0], symbol_position[1] + symbol_size),  # Bottom-left
  ]
  symbol_color = (0, 255, 0)
  draw.polygon(triangle_coordinates, fill=symbol_color)
  image.save(path)


  image = media.read_image(path)[:,:,:3]
  images.append(image)
  images.append(image)
  images.append(image)
  estim_path.append(path)


for path in rot1:
  image = media.read_image(path)[:,:,:3]
  images.append(image)
  images.append(image)

  estim_path.append(path)
estim_images = images

video = images
print(len(images))
media.write_video('logs/cutting_tissues_twice/nerf_onlytest_time_127000/ours.mp4', video, fps=7, qp=18)



