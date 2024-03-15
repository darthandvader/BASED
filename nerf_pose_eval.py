import os
import imageio
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from nerf_utils import *

from run_endonerf_helpers import *
from logging_nerf import *
from load_blender import load_blender_data
from load_llff import load_llff_data
from arg_parser import *

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


    print("depth maps ::: ", depth_maps)
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

def eval():
    parser = config_parser()
    args = parser.parse_args()
    args.ref_depths = True

    print("use_depth :: ", args.ref_depths)
    args.gt_fgmask = True ## do not comment for training

    times, render_times, poses, render_poses, i_train, i_test, hwf, near, far, depth_maps, images, masks = load_dataset(args)

    min_time, max_time = times[i_train[0]], times[i_train[-1]]
    assert min_time >= 0., "time must start at 0"
    assert max_time <= 1., "max time must be 1"

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    train_poses = torch.Tensor(np.array(poses[i_train]))
    render_poses = np.array(poses[i_test])
    render_times = np.array(times[i_test])

    basedir = args.basedir
    expname = args.expname
    nerf_dir = "nerf_only"
    if args.update_poses:
        nerf_dir = "nerf_and_pose"
    print("nerf_dir :: ", nerf_dir)
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
    
    gt_poses = train_poses
    training_poses = pose_net.get_w2c_poses()
    pose_error = calculate_pose_errors(training_poses, gt_poses)
    print("POSE ERROR :: ", pose_error)

    
if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    eval()