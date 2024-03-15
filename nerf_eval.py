import os
import imageio
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from nerf_utils import *
import cv2
from run_endonerf_helpers import *
from logging_nerf import *
from load_blender import load_blender_data
from load_llff import load_llff_data
from arg_parser import *
import open3d as o3d
import matplotlib.pyplot as plt
try:
    from apex import amp
except ImportError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = True

def load_dataset(args):
    images, masks, depth_maps, poses, times, bds, render_poses, render_times, i_test = load_llff_data(args, args.datadir, args.factor,
                                                                recenter=True, bd_factor=.75, spherify=args.spherify, fg_mask=args.gt_fgmask, use_depth=args.ref_depths,
                                                                render_path=args.llff_renderpath, davinci_endoscopic=args.davinci_endoscopic)


    # print("depth maps ::: ", depth_maps)
    hwf = poses[0,:3,-1] 
    poses = poses[:,:3,:4]
    print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

    if not isinstance(i_test, list):
        i_test = [i_test]

    if args.llffhold > 0 and not args.dataset == 'Hamlyn':
        print('Auto LLFF holdout,', args.llffhold)
        i_test = np.arange(images.shape[0])[1:-1:args.llffhold]

    i_val = i_test
    # i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in args.skip_frames)])  # use all frames for reconstruction
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val and i not in args.skip_frames)])  # leave out test/val frames

    print('DEFINING BOUNDS')
    print("I TEST ::", i_test)
    print("I TRAIN ::", i_train)
    close_depth, inf_depth = np.ndarray.min(bds) * .9, np.ndarray.max(bds) * 1.

    if args.no_ndc:
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.            
    else:
        near = 0.
        far = 1.
    print('NEAR FAR', near, far)

    if args.time_interval < 0:
        args.time_interval = 1 / (images.shape[0] - 1)

    return times, render_times, poses, render_poses, i_train, i_test, hwf, near, far, depth_maps, images, masks

def reconstruct_pointcloud(rgb_np, disp_np, hwf, vis_rgbd=True, verbose=True):
    
    depth_filter = (32, 64, 32)
    crop_left_size = 75
    depth_np = 1.0 / (disp_np + 1e-6)
    if crop_left_size > 0:
        rgb_np = rgb_np[:, crop_left_size:, :]
        depth_np = depth_np[:, crop_left_size:]

    if depth_filter is not None:
        depth_np = cv2.bilateralFilter(depth_np, depth_filter[0], depth_filter[1], depth_filter[2])

    if verbose:
        print('min disp:', disp_np.min(), 'max disp:', disp_np.max())
        print('min depth:', depth_np.min(), 'max depth:', depth_np.max())

    rgb_im = o3d.geometry.Image(rgb_np.astype(np.uint8))
    depth_im = o3d.geometry.Image(depth_np)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im, convert_rgb_to_intensity=False)

    if vis_rgbd:
        plt.subplot(1, 2, 1)
        plt.title('RGB image')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('Depth image')
        plt.imshow(rgbd_image.depth)
        plt.colorbar()
        plt.show()

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(hwf[1],hwf[0], hwf[2], hwf[2], hwf[1] / 2, hwf[0] / 2)
    )

    return pcd

def eval():
    parser = config_parser()
    args = parser.parse_args()

    times, render_times, poses, render_poses, i_train, i_test, hwf, near, far, depth_maps, images, masks = load_dataset(args)

    min_time, max_time = times[i_train[0]], times[i_train[-1]]
    assert min_time >= 0., "time must start at 0"
    assert max_time <= 1., "max time must be 1"

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    train_poses = torch.Tensor(np.array(poses[i_train]))

    identityt_gt_poses = train_poses
    render_poses = np.array(poses[i_test])
    render_times = np.array(times[i_test])

    basedir = args.basedir
    expname = args.expname
    nerf_dir = "nerf_only"
    if args.update_poses:
        nerf_dir = "nerf_and_pose"
    # print("nerf_dir :: ", nerf_dir)
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, nerf_model_extras = create_nerf(args, nerf_dir, args.lrate)
    global_step = start

    bds_dict = {
        'near' : near + 1e-6,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    render_times = torch.Tensor(render_times).to(device)

    if depth_maps is not None:
        close_depth, inf_depth = np.percentile(depth_maps, 3.0), np.percentile(depth_maps, 99.9)
    else:
        close_depth, inf_depth = near, far
    
    print('RENDER ONLY', train_poses.shape)
    pose_net = FirstTwoColunmnsPoseParameters(nbr_poses=train_poses.shape[0], 
                                                           initial_poses_w2c=train_poses, 
                                                           device='cuda')
    pose_dir = 'nerf_and_pose'
    ckpts = [os.path.join(args.basedir, args.expname, pose_dir, f) for f in sorted(os.listdir(os.path.join(args.basedir, args.expname, pose_dir))) if '.pth' in f and 'pose' in f]
    start_pose = 0
    if len(ckpts) != 0:
        print("pose net uploaded from checkpoint :: ", ckpts[-1])
        pose_net.load_state_dict(torch.load(ckpts[-1]))

    with torch.no_grad():
        images = images[i_test]
        if masks is not None:
            masks = masks[i_test]
        save_gt = True
        if args.test_time_optim:
            testsavedir = os.path.join(basedir, expname, nerf_dir + 'test_time_{:06d}'.format(start))
        else:
            testsavedir = os.path.join(basedir, expname, nerf_dir + '_{:06d}'.format(start))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        rgbs, disps = render_path(args, pose_net, identityt_gt_poses, depth_maps, render_poses, render_times, hwf, args.chunk, render_kwargs_test, gt_imgs=images, gt_masks=masks,
                                savedir=testsavedir, render_factor=args.render_factor, save_also_gt=save_gt, save_depth=True, near_far=(near, far))
        # rgbs, _ = render_novel_views(render_poses, render_times, hwf, args.chunk, render_kwargs_test, gt_imgs=images, gt_masks=masks,
        #                           savedir=testsavedir, render_factor=args.render_factor, save_also_gt=save_gt, save_depth=True, 
        #                           near_far=(close_depth, inf_depth))
        print('Done rendering', testsavedir)
        pcds = []
        for rgb,disp in zip(rgbs,disps):
            pcd = reconstruct_pointcloud(rgb, disp, hwf)
            pcds.append(pcd)
        print('Saving point clouds...')

        pc_dir = os.path.join(testsavedir,"reconstructed_pcds")
        for i, pcd in enumerate(pcds):
            fn = os.path.join(pc_dir, f"frame_{i:06d}_pc.ply")
            o3d.io.write_point_cloud(fn, pcd)
        print('Point clouds saved to', pc_dir)
        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=args.video_fps, quality=8)


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    eval()