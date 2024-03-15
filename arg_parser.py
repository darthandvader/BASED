import os
import imageio
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from nerf_utils import *

from run_endonerf_helpers import *

from load_blender import load_blender_data
from load_llff import load_llff_data


try:
    from apex import amp
except ImportError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = True

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')
    parser.add_argument("--dataset", type=str, default='endonerf', 
                        help='input data directory')
    parser.add_argument("--H", type=float, default=0.0, 
                        help='height')
    parser.add_argument("--W", type=float, default=0.0, 
                        help='weight')
    parser.add_argument("--focal", type=float, default=0.0, 
                        help='focal')
    # training options
    parser.add_argument("--nerf_type", type=str, default="original",
                        help='nerf network type')
    parser.add_argument("--N_iter", type=int, default=100000,
                        help='num training iterations')
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--do_half_precision", action='store_true',
                        help='do half precision training and inference')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_nerf_simul", type=float, default=3e-3, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--update_poses",  action='store_true',
                        help='updates poses and nerf simultaneously')
    parser.add_argument("--test_time_optim",  action='store_true',
                        help='Should GT poses be used at test time, or should the model optimize poses at test time')
    parser.add_argument("--ref_depths",  default=False,
                        help='Should GT poses be used at test time, or should the model optimize poses at test time')
    
    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--not_zero_canonical", action='store_true',
                        help='if set zero time is not the canonic space')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--use_two_models_for_fine", action='store_true',
                        help='use two models for fine results')
                        
    parser.add_argument("--time_window_size", type=int, default=3, 
                        help='the size of time window in recurrent temporal nerf')
    parser.add_argument("--time_interval", type=float, default=-1, 
                        help='the time interval between two adjacent frames')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training trick options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_iters_time", type=int, default=0,
                        help='number of steps to train on central time')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')
    parser.add_argument("--add_tv_loss", action='store_true',
                        help='evaluate tv loss')
    parser.add_argument("--tv_loss_weight", type=float,
                        default=1.e-4, help='weight of tv loss')
    parser.add_argument("--gt_fgmask", action='store_true',
                        help='Should GT (or reference masks) be used? Should not use this if no masks data is available')
    parser.add_argument("--no_mask_raycast", action='store_true',
                        help='disable tool mask-guided ray-casting')
    parser.add_argument("--mask_loss", action='store_true',
                        help='enable erasing loss for masked pixels')
    parser.add_argument("--novel_views", action='store_true',
                        help='Should novel view rendering happen during test time?')
    parser.add_argument("--no_depth_sampling", action='store_true',
                        help='disable depth-guided ray sampling?')
    parser.add_argument("--depth_sampling_sigma", type=float, default=5.0,
                        help='std of depth-guided sampling')
    parser.add_argument("--depth_loss_weight", type=float, default=1.0,
                        help='weight of depth loss')
    parser.add_argument("--no_depth_refine", action='store_true',
                        help='disable depth refinement') 
    parser.add_argument("--depth_refine_period", type=int, default=1000,
                        help='number of iters to refine depth maps') 
    parser.add_argument("--depth_refine_rounds", type=int, default=4,
                        help='number of rounds of depth map refinement') 
    parser.add_argument("--depth_refine_quantile", type=float, default=0.3,
                        help='proportion of pixels to be updated during depth refinement')           


    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=2,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--davinci_endoscopic", action='store_true',
                        help='is Da Vinci endoscopic surgical fields?')
    parser.add_argument("--skip_frames", nargs='+', type=int, default=[], 
                        help='skip frames for training')

    ## deepvoxels flags (unused)
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags (unused)
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')
    parser.add_argument("--llff_renderpath", type=str, default='spiral', 
                        help='options: spiral, fixidentity, zoom')
                                                
    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1000,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=10000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=100, 
                        help='frequency of weight ckpt saving') ## hardcodedd
    parser.add_argument("--i_testset", type=int, default=200000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=200000,
                        help='frequency of render_poses video saving')
    parser.add_argument("--video_fps",  type=int, default=30,
                        help='FPS of render_poses video')
    

    return parser