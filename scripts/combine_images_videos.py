import glob
import mediapy as media
from natsort import natsorted
videos = {}
gt_paths = []
paths = glob.glob("logs/cutting_tissues_twice/nerf_onlytest_time_127000/gt/*.rgb.png")
paths = natsorted(paths)[1:]
paths1 = paths[:52]
paths2 = paths[52:104]
paths3 = paths[104:156]
rot1 = glob.glob("logs/cutting_tissues_twice/nerf_onlytest_time_127000/novel/pose_52_*.rgb.png")
rot1 = natsorted(rot1)[1:]

rot2 = glob.glob("logs/cutting_tissues_twice/nerf_onlytest_time_127000/novel/pose_104_*.rgb.png")
rot2 = natsorted(rot1)[1:]
# print(paths)
images = []
for path in paths1:
  image = media.read_image(path)[:,:,:3]
  images.append(image)
  gt_paths.append(path)
for path in rot1:
  image = media.read_image(paths[51])[:,:,:3]
  images.append(image)
  images.append(image)
  images.append(image)
  gt_paths.append(paths[51])
  gt_paths.append(paths[51])
  gt_paths.append(paths[51])
  gt_paths.append(paths[51])
  gt_paths.append(paths[51])
for path in paths2:
  image = media.read_image(path)[:,:,:3]
  images.append(image)
  gt_paths.append(path)
for path in rot2:
  image = media.read_image(paths[103])[:,:,:3]
  images.append(image)
  images.append(image)
  images.append(image)
  gt_paths.append(paths[103])
  gt_paths.append(paths[103])
  gt_paths.append(paths[103])
  gt_paths.append(paths[103])
  gt_paths.append(paths[103])
for path in paths3:
  image = media.read_image(path)[:,:,:3]
  images.append(image)
  gt_paths.append(path)
gt_images = images
print(len(images))


videos = {}
estim_path = []
paths = glob.glob("logs/cutting_tissues_twice/nerf_onlytest_time_127000/estim/*.rgb.png")
paths = natsorted(paths)[1:]
paths1 = paths[:52]
paths2 = paths[52:104]
paths3 = paths[104:156]
rot1 = glob.glob("logs/cutting_tissues_twice/nerf_onlytest_time_127000/novel/pose_52_*.rgb.png")
rot1 = natsorted(rot1)[1:]

rot2 = glob.glob("logs/cutting_tissues_twice/nerf_onlytest_time_127000/novel/pose_104_*.rgb.png")
rot2 = natsorted(rot1)[1:]
# print(paths)
images = []
for path in paths1:
  image = media.read_image(path)[:,:,:3]
  images.append(image)
  estim_path.append(path)
for path in rot1:
  image = media.read_image(path)[:,:,:3]
  images.append(image)
  images.append(image)
  images.append(image)
  estim_path.append(path)
  estim_path.append(path)
  estim_path.append(path)
  estim_path.append(path)
  estim_path.append(path)
for path in paths2:
  image = media.read_image(path)[:,:,:3]
  images.append(image)
  estim_path.append(path)
for path in rot2:
  image = media.read_image(path)[:,:,:3]
  images.append(image)
  images.append(image)
  images.append(image)
  estim_path.append(path)
  estim_path.append(path)
  estim_path.append(path)
  estim_path.append(path)
  estim_path.append(path)
for path in paths3:
  image = media.read_image(path)[:,:,:3]
  images.append(image)
  estim_path.append(path)
estim_images = images


endo_paths = glob.glob("logs/cutting_tissues_twice/nerf_onlytest_time_127000/combined_endonerf/*")
endo_paths = natsorted(endo_paths)

print(len(gt_paths), len(endo_paths), len(estim_path))

import sys
from PIL import Image

for i,(gt,endo,est) in enumerate(zip(gt_paths, endo_paths, estim_path)):
    print(gt,est)
    images = [Image.open(x) for x in [gt,endo,est]]

    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save('logs/cutting_tissues_twice/nerf_onlytest_time_127000/combined/' + str(i) + '.jpg')

print(len(images))
media.write_video('logs/cutting_tissues_twice/nerf_onlytest_time_127000/all_combined.mp4', video, fps=2, qp=18)