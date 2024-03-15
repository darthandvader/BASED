import numpy as np
from matplotlib import pyplot as plt

depth_npy_file = np.load('logs/rectified06/nerf_onlytest_time_107000/estim/000.depth.npy',allow_pickle=True)
print(depth_npy_file.shape)
plt.imshow(depth_npy_file)
plt.savefig('logs/rectified06/nerf_onlytest_time_107000/estim/000_depth.png')