import glob
import mediapy as media
from natsort import natsorted
videos = {}
paths = glob.glob("logs/rectified18_1/nerf_onlytest_time_120000/gt/*.rgb.png")
paths = natsorted(paths)[20:]

images = []
for path in paths:
  image = media.read_image(path)[:,:,:3]
  images.append(image)

video = images
print(len(images))
media.write_video('logs/rectified18_1/nerf_onlytest_time_120000/gt.mp4', video, fps=5, qp=18)