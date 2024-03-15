import os
import imageio
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from camera import *
import torchvision
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


def create_nerf(args, nerf_dir, lrate):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, 3, args.i_embed)
    embedtime_fn, input_ch_time = get_embedder(args.multires, 1, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, 3, args.i_embed)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [args.netdepth // 2]
    model = NeRF.get_by_name(args.nerf_type, D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, input_ch_time=input_ch_time,
                 use_viewdirs=args.use_viewdirs, embed_fn=embed_fn, embedtime_fn=embedtime_fn,
                 zero_canonical=not args.not_zero_canonical, time_window_size=args.time_window_size, time_interval=args.time_interval).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.use_two_models_for_fine:
        model_fine = NeRF.get_by_name(args.nerf_type, D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, input_ch_time=input_ch_time,
                          use_viewdirs=args.use_viewdirs, embed_fn=embed_fn, embedtime_fn=embedtime_fn,
                          zero_canonical=not args.not_zero_canonical, time_window_size=args.time_window_size, time_interval=args.time_interval).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, ts, network_fn, progress : run_network(inputs, viewdirs, ts, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                embedtime_fn=embedtime_fn,
                                                                progress=progress,
                                                                netchunk=args.netchunk,
                                                                embd_time_discr=args.nerf_type!="temporal")

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9, 0.999))

    if args.do_half_precision:
        print("Run model at half precision")
        if model_fine is not None:
            [model, model_fine], optimizers = amp.initialize([model, model_fine], optimizer, opt_level='O1')
        else:
            model, optimizers = amp.initialize(model, optimizer, opt_level='O1')

    # Extras
    extras = {
        'depth_maps': None,
        'ray_importance_maps': None
    }

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname,nerf_dir, f) for f in sorted(os.listdir(os.path.join(basedir, expname, nerf_dir))) if 'tar' in f]

    if len(ckpts) == 0:
        ckpts = [os.path.join(basedir, expname,"nerf_and_pose", f) for f in sorted(os.listdir(os.path.join(basedir, expname, "nerf_and_pose"))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step'] + 1
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        if args.do_half_precision:
            amp.load_state_dict(ckpt['amp'])

        # Load extras
        if 'depth_maps' in ckpt:
            extras['depth_maps'] = ckpt['depth_maps']
        if 'ray_importance_maps' in ckpt:
            extras['ray_importance_maps'] = ckpt['ray_importance_maps']


    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine': model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'use_two_models_for_fine' : args.use_two_models_for_fine,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    if args.ref_depths and not args.no_depth_sampling:
        render_kwargs_train['use_depth'] = True

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, extras

def render_rays(progress,
                ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                z_vals=None,
                use_two_models_for_fine=False,
                use_depth=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 9 else None
    bounds = torch.reshape(ray_batch[...,6:9], [-1,1,3])
    near, far, frame_time = bounds[...,0], bounds[...,1], bounds[...,2] # [-1,1]
    z_samples = None
    rgb_map_0, disp_map_0, acc_map_0, position_delta_0 = None, None, None, None

    if z_vals is None:
        if not use_depth:
            t_vals = torch.linspace(0., 1., steps=N_samples)
            if not lindisp:
                z_vals = near * (1.-t_vals) + far * (t_vals)
            else:
                z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

            z_vals = z_vals.expand([N_rays, N_samples])
        else:
            mean = near.expand([N_rays, N_samples])
            std = far.expand([N_rays, N_samples])
            z_vals, _ = torch.sort(torch.normal(mean, std), dim=1)

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


        # if (torch.isnan(pts).any() or torch.isinf(pts).any()) and DEBUG:
        #     print(f"! [Numerical Error] pts contains nan or inf.", flush=True)

        # if (torch.isnan(rays_o).any() or torch.isinf(rays_o).any()) and DEBUG:
        #     print(f"! [Numerical Error] rays_o contains nan or inf.", flush=True)

        # if (torch.isnan(rays_d).any() or torch.isinf(rays_d).any()) and DEBUG:
        #     print(f"! [Numerical Error] rays_d contains nan or inf.", flush=True)

        # if (torch.isnan(z_vals).any() or torch.isinf(z_vals).any()) and DEBUG:
        #     print(f"! [Numerical Error] z_vals contains nan or inf.", flush=True)


        if N_importance <= 0:
            raw, position_delta = network_query_fn(pts, viewdirs, frame_time, network_fn)
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

        else:
            if use_two_models_for_fine:
                raw, position_delta_0 = network_query_fn(pts, viewdirs, frame_time, network_fn)
                rgb_map_0, disp_map_0, acc_map_0, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

            else:
                with torch.no_grad():
                    raw, _ = network_query_fn(pts, viewdirs, frame_time, network_fn, progress)
                    _, _, _, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = importance_sampling_ray(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
    run_fn = network_fn if network_fine is None else network_fine
    raw, position_delta = network_query_fn(pts, viewdirs, frame_time, run_fn, progress)
    rgb_map, disp_map, acc_map, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'z_vals' : z_vals,
           'position_delta' : position_delta}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        if rgb_map_0 is not None:
            ret['rgb0'] = rgb_map_0
        if disp_map_0 is not None:
            ret['disp0'] = disp_map_0
        if acc_map_0 is not None:
            ret['acc0'] = acc_map_0
        if position_delta_0 is not None:
            ret['position_delta_0'] = position_delta_0
        if z_samples is not None:
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    # for k in ret:
    #     if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
    #         print(f"! [Numerical Error] {k} contains nan or inf.", flush=True)

    return ret

def batchify_rays(progress, rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """

    # if (torch.isnan(rays_flat).any() or torch.isinf(rays_flat).any()) and DEBUG:
    #     print(f"! [Numerical Error] rays_flat contains nan or inf.", flush=True)
    
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(progress, rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs_pos, inputs_time, progress):
        num_batches = inputs_pos.shape[0]

        out_list = []
        dx_list = []
        for i in range(0, num_batches, chunk):
            out, dx = fn(inputs_pos[i:i+chunk], [inputs_time[0][i:i+chunk], inputs_time[1][i:i+chunk]], progress)
            out_list += [out]
            dx_list += [dx]

        return torch.cat(out_list, 0), torch.cat(dx_list, 0)
    return ret

def run_network(inputs, viewdirs, frame_time, fn, embed_fn, embeddirs_fn, embedtime_fn, progress, netchunk=1024*64,
                embd_time_discr=True):
    """Prepares inputs and applies network 'fn'.
    inputs: N_rays x N_points_per_ray x 3
    viewdirs: N_rays x 3
    frame_time: N_rays x 1
    """

    assert len(torch.unique(frame_time)) == 1, "Only accepts all points from same time"
    cur_time = torch.unique(frame_time)[0]

    # embed position
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat, progress)

    # embed time
    if embd_time_discr:
        B, N, _ = inputs.shape
        input_frame_time = frame_time[:, None].expand([B, N, 1])
        input_frame_time_flat = torch.reshape(input_frame_time, [-1, 1])
        embedded_time = embedtime_fn(input_frame_time_flat, progress)
        embedded_times = [embedded_time, embedded_time]

    else:
        assert NotImplementedError

    # embed views
    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat, progress)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat, position_delta_flat = batchify(fn, netchunk)(embedded, embedded_times, progress)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    position_delta = torch.reshape(position_delta_flat, list(inputs.shape[:-1]) + [position_delta_flat.shape[-1]])

    return outputs, position_delta

def render(progress, H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,

                  near=0., far=1., frame_time=None,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    # if (torch.isnan(rays_o).any() or torch.isinf(rays_o).any()) and DEBUG:
    #     print(f"! [Numerical Error] rays_o in render 1 contains nan or inf.", flush=True)
    # if (torch.isnan(rays_d).any() or torch.isinf(rays_d).any()) and DEBUG:
    #     print(f"! [Numerical Error] rays_d in render 1 contains nan or inf.", flush=True)

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / (torch.norm(viewdirs, dim=-1, keepdim=True) + 1e-6)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d) ##shreya check TODO

    # if (torch.isnan(rays_o).any() or torch.isinf(rays_o).any()) and DEBUG:
    #     print(f"! [Numerical Error] rays_o in render 2 contains nan or inf.", flush=True)
    # if (torch.isnan(rays_d).any() or torch.isinf(rays_d).any()) and DEBUG:
    #     print(f"! [Numerical Error] rays_d in render 2 contains nan or inf.", flush=True)
    

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    if 'use_depth' in kwargs and kwargs['use_depth']:
        # near is the mean of depth, far is the std of depth
        near = near.unsqueeze(0).reshape(-1, 1)
        far = far * torch.ones_like(near)
    else:
        near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    frame_time = frame_time * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far, frame_time], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(progress, rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

@torch.enable_grad()
def evaluate_test_time_photometric_optim(args, train_net, H, W, focal, c2w, frame_time, mask, ray_importance_map, render_kwargs_train, target, i):

    pose_net = FirstTwoColunmnsPoseParameters(nbr_poses=1, 
                                                           initial_poses_w2c=c2w.unsqueeze(0), 
                                                           device='cuda')
    optimizer1 = getattr(torch.optim,'Adam')
    optimizer_pose = optimizer1([dict(params=pose_net.parameters(), lr=5e-4)])
    scheduler = getattr(torch.optim.lr_scheduler,'ExponentialLR')
    lr_pose = 5e-4
    max_iter = 100
    lr_pose_end = 1.e-7 
    gamma = (lr_pose_end/lr_pose)**(1./max_iter)
    kwargs = {'gamma':gamma }
    scheduler_pose = scheduler(optimizer_pose,**kwargs)

    training_poses = train_net.get_w2c_poses() 
    testing_pose = pose_net.get_w2c_poses() 

    N_rand = 4096
    
    from tqdm  import tqdm
    for it in tqdm(range(max_iter)):
        optimizer_pose.zero_grad()
        testing_pose = pose_net.get_w2c_poses() 
        # pose_refine_test = lie.se3_to_SE3(se3_refine_test)
        pose_refine_test = testing_pose[0]
        # test_pose = pose.compose([pose_refine_test,training_poses])
        test_pose = pose_refine_test
        
        rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(test_pose))
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1) 
        coords = torch.reshape(coords, [-1,2])
        select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False) ## without using masks
        # tmp = ray_importance_map[coords[:, 0].long(), coords[:, 1].long()]
        # select_inds, _, cdf = importance_sampling_coords(ray_importance_map[coords[:, 0].long(), coords[:, 1].long()].unsqueeze(0), N_rand)
        # select_inds = torch.max(torch.zeros_like(select_inds), select_inds)
        # select_inds = torch.min((coords.shape[0] - 1) * torch.ones_like(select_inds), select_inds)
        # select_inds = select_inds.squeeze(0)

        select_coords = coords[select_inds].long()  # (N_rand, 2)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        batch_rays = torch.stack([rays_o, rays_d], 0)
        select_coords = select_coords.cpu().numpy()
        target_s = torch.Tensor(target[select_coords[:, 0], select_coords[:, 1]])
        mask_s = None
        rgb, disp, acc, extras = render(1.0, H, W, focal, chunk=args.chunk, rays=batch_rays, frame_time=frame_time,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        img_loss = img2mse(rgb, target_s)
        img_loss.backward(retain_graph=True)
        optimizer_pose.step()
        # scheduler_pose.step()
    return pose_net.get_w2c_poses() [0]

def calculate_pose_errors(training_poses, pose_GT):
    R_aligned, t_aligned = training_poses.split([3, 1], dim=-1)
    print("R_aligned, t_aligned", R_aligned.shape, t_aligned.shape)
    R_GT, t_GT = pose_GT.split([3, 1], dim=-1)
    print("R_GT, t_GT", R_GT.shape, t_GT.shape)
    R_error = rotation_distance(R_aligned, R_GT)
    t_error = (t_aligned-t_GT)[..., 0].norm(dim=-1)
    R_error = R_error.mean().item()
    t_error = t_error.mean().item()
    error = edict(R=R_error, t=t_error)
    return error

def positional_encoding(self, opt: Dict[str, Any], input: torch.Tensor, 
                            embedder_fn: Callable[[Dict[str, Any], torch.Tensor, int], torch.Tensor], 
                            L: int) -> torch.Tensor: # [B,...,N]
        """Apply the coarse-to-fine positional encoding strategy of BARF. 

        Args:
            opt (edict): settings
            input (torch.Tensor): shaps is (B, ..., C) where C is channel dimension
            embedder_fn (function): positional encoding function
            L (int): Number of frequency basis
        returns:
            positional encoding
        """
        shape = input.shape
        input_enc = embedder_fn(opt, input, L) # [B,...,2NL]

        # [B,...,2NL]
        # coarse-to-fine: smoothly mask positional encoding for BARF
        # the progress could also be fixed but then we finetune without cf_pe, then it is just not updated 
        if opt.barf_c2f is not None:
            # set weights for different frequency bands
            start,end = opt.barf_c2f
            alpha = (self.progress.data-start)/(end-start)*L
            k = torch.arange(L,dtype=torch.float32,device=input.device)
            weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2

            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1,L)*weight).view(*shape)
        return input_enc

def get_ray_importance_maps(masks):
    if masks is not None:
        masks = torch.Tensor(masks).to(device)
        ray_importance_maps = ray_sampling_importance_from_masks(masks)
    else:
        ray_importance_maps = None
    return ray_importance_maps

def render_novel_views(c2w, max_trans):
    spiral_poses = getSpiralPoses(c2w, max_trans)
    
    return None

def getSpiralPoses(c2w,max_trans):
    spiral_poses = []
    c2w = c2w.cpu().detach().numpy()
    # Rendering teaser. Add translation.
    z_trans_list = result = np.linspace(0, 0.5, 75)
    for i in range(75):
        x_trans = max_trans * 0.5 * np.sin(1.0 * np.pi * float(i) / float(10)) * 2.0
        y_trans = (
            max_trans
            * 0.5
            * (np.cos(2.0 * np.pi * float(i) / float(10)) - 1.0)
            * 1.0
            / 3.0
        )
        z_trans = z_trans_list[i]

        i_pose = np.concatenate(
            [
                np.concatenate(
                    [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]],
                    axis=1,
                ),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
            ],
            axis=0,
        )

        i_pose = np.linalg.inv(i_pose)

        ref_pose = np.concatenate(
            [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0
        )

        render_pose = np.dot(ref_pose, i_pose)
        # output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
        spiral_poses.append(render_pose[:3, :])
    spiral_poses = np.stack(spiral_poses, 0)[:, :3]
    return torch.tensor(spiral_poses)

def render_path(args, pose_net, identity_gt_poses, depth_maps, render_poses, render_times, hwf, chunk, render_kwargs, gt_imgs=None, gt_masks=None, savedir=None,
                render_factor=0, save_also_gt=False, i_offset=0, save_depth=False, near_far=(0, 1)):

    H, W, focal = hwf
    print("umm")
    novel = args.novel_views
    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    if savedir is not None:
        save_dir_estim = os.path.join(savedir, "estim")
        save_dir_gt = os.path.join(savedir, "gt")
        save_dir_gt_masks = os.path.join(savedir, "gt_masks")
        novel_view_dir = os.path.join(savedir, "novel")
        if not os.path.exists(save_dir_estim):
            os.makedirs(save_dir_estim)
        if save_also_gt and not os.path.exists(save_dir_gt):
            os.makedirs(save_dir_gt)
        if save_also_gt and not os.path.exists(save_dir_gt_masks):
            os.makedirs(save_dir_gt_masks)
        if save_also_gt and not os.path.exists(novel_view_dir):
            os.makedirs(novel_view_dir)

    rgbs = []
    disps = []
    test_poses = []
    training_poses = pose_net.get_w2c_poses()
    plotting_poses = plot_training_poses(training_poses, -1)
    pose_error = calculate_pose_errors(training_poses, identity_gt_poses)
    print("pose error :: ", pose_error)

    if args.test_time_optim:
        torchvision.utils.save_image(plotting_poses['training poses'], os.path.join(save_dir_estim,'training_poses.png'))

    ray_importance_maps = get_ray_importance_maps(gt_masks)
    # sc = near_far[0] * 0.75
    max_trans = 24.0/focal
    if gt_masks is None:
        print("here1")
        for i, (c2w, frame_time) in enumerate(zip(tqdm(render_poses), render_times)):
            # if os.path.exists(os.path.join(save_dir_estim, '{:03d}.rgb.png'.format(i+i_offset))):
            #     continue
            if args.test_time_optim:
                c2w = evaluate_test_time_photometric_optim(args, pose_net, H, W, focal, c2w, frame_time, None, None, render_kwargs, gt_imgs[i], i)
                print("rendered pose :: ", c2w)
                test_poses.append(c2w.cpu().detach().numpy())
            if novel:
                spiral_views = getSpiralPoses(c2w, max_trans)
                continue
            rgb, disp, acc, _ = render(1.0, H, W, focal, chunk=chunk, c2w=c2w[:3,:4], frame_time=frame_time, **render_kwargs)
            rgbs.append(rgb.cpu().numpy())
            disps.append(disp.cpu().numpy())

            if savedir is not None:
                rgb8_estim = to8b(rgbs[-1])
                filename = os.path.join(save_dir_estim, '{:03d}.rgb.png'.format(i+i_offset))
                imageio.imwrite(filename, rgb8_estim)
                
                if save_also_gt:
                    rgb8_gt = to8b(gt_imgs[i])
                    filename = os.path.join(save_dir_gt, '{:03d}.rgb.png'.format(i+i_offset))
                    imageio.imwrite(filename, rgb8_gt)

                    rgb8_gt = to8b(depth_maps[i])
                    filename = os.path.join(save_dir_gt, '{:03d}.depth.png'.format(i+i_offset))
                    imageio.imwrite(filename, rgb8_gt)

                    if gt_masks is not None:
                        rgb8_gt_mask = to8b(gt_masks[i])
                        arr = to8b(gt_masks[i])
                        print(type(rgb8_gt_mask),rgb8_gt_mask.shape)
                        rgb8_gt_mask[np.where(arr == 1.0)] = 0.0
                        rgb8_gt_mask[np.where(arr == 0.0)] = 1.0
                        print(type(rgb8_gt_mask),rgb8_gt_mask.shape)
                        filename = os.path.join(save_dir_gt_masks, '{:03d}.rgb.png'.format(i+i_offset))
                        imageio.imwrite(filename, rgb8_gt_mask)
                
                if save_depth:
                    depth_estim = (1.0 / (disps[-1] + 1e-6)) * (near_far[1] - near_far[0])
                    filename = os.path.join(save_dir_estim, '{:03d}.depth.npy'.format(i+i_offset))
                    np.save(filename, depth_estim)
    else:
        print("here2")
        for i, (c2w, frame_time, mask, ray_importance_map) in tqdm(enumerate(zip(tqdm(render_poses), render_times, gt_masks, ray_importance_maps))):
            
            # if os.path.exists(os.path.join(save_dir_estim, '{:03d}.rgb.png'.format(i+i_offset))):
            #     continue
            if args.test_time_optim:
                c2w = evaluate_test_time_photometric_optim(args, pose_net, H, W, focal, c2w, frame_time, mask, ray_importance_map, render_kwargs, gt_imgs[i], i)
                print("rendered pose :: ", c2w)
                test_poses.append(c2w.cpu().detach().numpy())
            print("novel views :: ", novel)
            if novel:
                spiral_views = getSpiralPoses(c2w, max_trans)
                print("spiral_views :: ", spiral_views.shape)
                for j,spiral_c2w in enumerate(spiral_views):
                    rgb, disp, acc, _ = render(1,H, W, focal, chunk=chunk, c2w=spiral_c2w[:3,:4], frame_time=frame_time, **render_kwargs)
                    filename = os.path.join(novel_view_dir, 'pose_'+str(i)+'_'+'{:03d}.rgb.png'.format(j+i_offset))
                    imageio.imwrite(filename, to8b(rgb.cpu().numpy()))
                continue
            rgb, disp, acc, depth = render(1,H, W, focal, chunk=chunk, c2w=c2w[:3,:4], frame_time=frame_time, **render_kwargs)
            rgbs.append(rgb.cpu().numpy())
            disps.append(disp.cpu().numpy())

            if savedir is not None:
                rgb8_estim = to8b(rgbs[-1])
                filename = os.path.join(save_dir_estim, '{:03d}.rgb.png'.format(i+i_offset))
                imageio.imwrite(filename, rgb8_estim)
                
                if save_also_gt:
                    rgb8_gt = to8b(gt_imgs[i])
                    filename = os.path.join(save_dir_gt, '{:03d}.rgb.png'.format(i+i_offset))
                    imageio.imwrite(filename, rgb8_gt)

                    rgb8_gt_mask = to8b(gt_masks[i])
                    arr = to8b(gt_masks[i])
                    print(type(rgb8_gt_mask),rgb8_gt_mask.shape)
                    rgb8_gt_mask[np.where(arr == 1.0)] = 0.0
                    rgb8_gt_mask[np.where(arr == 0.0)] = 1.0
                    print(type(rgb8_gt_mask),rgb8_gt_mask.shape)
                    filename = os.path.join(save_dir_gt_masks, '{:03d}.rgb.png'.format(i+i_offset))
                    imageio.imwrite(filename, rgb8_gt_mask)
            
                if save_depth:
                    depth_estim = (1.0 / (disps[-1] + 1e-6)) ##* (near_far[1] - near_far[0])
                    filename = os.path.join(save_dir_estim, '{:03d}.depth.npy'.format(i+i_offset))
                    np.save(filename, depth_estim)
    plotting_poses = plot_training_poses(torch.tensor(test_poses), -1)
    if args.test_time_optim:
        torchvision.utils.save_image(plotting_poses['training poses'], os.path.join(save_dir_estim,'test_time_optimised_poses.png'))
        pose_errors = calculate_pose_errors(torch.tensor(test_poses), render_poses)
        print("TEST POSE ERRORS :: ", pose_errors)
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    depth_maps = 1.0 / (disps + 1e-6)
    close_depth, inf_depth = np.percentile(depth_maps, 3.0), np.percentile(depth_maps, 99.0)
    if save_depth:
        for i, depth in enumerate(depth_maps):
            depth8_estim = to8b(depth / ((inf_depth - close_depth) + 1e-6))
            filename = os.path.join(save_dir_estim, '{:03d}.depth.png'.format(i+i_offset))
            imageio.imwrite(filename, depth8_estim)

    return rgbs, disps

def render_path_original(args, pose_net, gt_identity_poses, render_poses, render_times, gt_depths, hwf, chunk, render_kwargs, gt_imgs=None, gt_masks=None, savedir=None,
                render_factor=0, save_also_gt=False, i_offset=0, save_depth=False, near_far=(0, 1)):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    if savedir is not None:
        save_dir_estim = os.path.join(savedir, "estim")
        save_dir_gt = os.path.join(savedir, "gt")
        save_dir_gt_masks = os.path.join(savedir, "gt_masks")
        if not os.path.exists(save_dir_estim):
            os.makedirs(save_dir_estim)
        if save_also_gt and not os.path.exists(save_dir_gt):
            os.makedirs(save_dir_gt)
        if save_also_gt and not os.path.exists(save_dir_gt_masks):
            os.makedirs(save_dir_gt_masks)

    rgbs = []
    disps = []
    test_poses = []
    plotting_poses = plot_training_poses(pose_net.get_w2c_poses(), -1)
    pose_errors = calculate_pose_errors(pose_net.get_w2c_poses(), gt_identity_poses)
    print("POSE ERRORS!!!!! ", pose_errors)
    if args.test_time_optim:
        torchvision.utils.save_image(plotting_poses['training poses'], os.path.join(save_dir_estim,'training_poses.png'))

    ray_importance_maps = get_ray_importance_maps(gt_masks)

    for i, (c2w, frame_time, mask, ray_importance_map) in enumerate(zip(tqdm(render_poses), render_times, gt_masks, ray_importance_maps)):
        if os.path.exists(os.path.join(save_dir_estim, '{:03d}.rgb.png'.format(i+i_offset))):
            continue
        if args.test_time_optim:
            c2w = evaluate_test_time_photometric_optim(args, pose_net, H, W, focal, c2w, frame_time, mask, ray_importance_map, render_kwargs, gt_imgs[i], i)
            print("rendered pose :: ", c2w)
            test_poses.append(c2w.cpu().detach().numpy())
        rgb, disp, acc, _ = render(1, H, W, focal, chunk=chunk, c2w=c2w[:3,:4], frame_time=frame_time, **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if savedir is not None:
            rgb8_estim = to8b(rgbs[-1])
            filename = os.path.join(save_dir_estim, '{:03d}.rgb.png'.format(i+i_offset))
            imageio.imwrite(filename, rgb8_estim)
            
            if save_also_gt:
                rgb8_gt = to8b(gt_imgs[i])
                filename = os.path.join(save_dir_gt, '{:03d}.rgb.png'.format(i+i_offset))
                imageio.imwrite(filename, rgb8_gt)

                rgb8_gt_mask = to8b(gt_masks[i])
                arr = to8b(gt_masks[i])
                print(type(rgb8_gt_mask),rgb8_gt_mask.shape)
                rgb8_gt_mask[np.where(arr == 1.0)] = 0.0
                rgb8_gt_mask[np.where(arr == 0.0)] = 1.0
                print(type(rgb8_gt_mask),rgb8_gt_mask.shape)
                filename = os.path.join(save_dir_gt_masks, '{:03d}.rgb.png'.format(i+i_offset))
                imageio.imwrite(filename, rgb8_gt_mask)
                
                filename = os.path.join(save_dir_gt, '{:03d}.depth.npy'.format(i+i_offset))
                np.save(filename, gt_depths[i])
            
            if save_depth:
                depth_estim = (1.0 / (disps[-1] + 1e-6)) * (near_far[1] - near_far[0])
                filename = os.path.join(save_dir_estim, '{:03d}.depth.npy'.format(i+i_offset))
                np.save(filename, depth_estim)
    plotting_poses = plot_training_poses(torch.tensor(test_poses), -1)
    if args.test_time_optim:
        torchvision.utils.save_image(plotting_poses['training poses'], os.path.join(save_dir_estim,'test_time_optimised_poses.png'))
        pose_errors = calculate_pose_errors(torch.tensor(test_poses), render_poses)
        print("TEST POSE ERRORS :: ", pose_errors)
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    depth_maps = 1.0 / (disps + 1e-6)
    close_depth, inf_depth = np.percentile(depth_maps, 3.0), np.percentile(depth_maps, 99.0)
    if save_depth:
        for i, depth in enumerate(depth_maps):
            depth8_estim = to8b(depth / ((inf_depth - close_depth) + 1e-6))
            filename = os.path.join(save_dir_estim, '{:03d}.depth.png'.format(i+i_offset))
            imageio.imwrite(filename, depth8_estim)

    return rgbs, disps

def render_path_gpu(render_poses, render_times, hwf, chunk, render_kwargs, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    for i, (c2w, frame_time) in enumerate(zip(tqdm(render_poses), render_times)):
        rgb, disp, _, _ = render(1.0,H, W, focal, chunk=chunk, c2w=c2w[:3,:4], frame_time=frame_time, **render_kwargs)
        rgbs.append(rgb)
        disps.append(disp)

    rgbs = torch.stack(rgbs, 0)
    disps = torch.stack(disps, 0)

    return rgbs, disps

import vis_rendering as util_vis
def plot_training_poses(poses, iter):
    plotting_dict = {}
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20,10))
    pose_vis = util_vis.plot_save_poses(fig,poses,iter)
    pose_vis = torch.from_numpy(pose_vis.astype(np.float32)/255.).permute(2, 0, 1)
    plotting_dict['training poses'] = pose_vis
    return plotting_dict
    
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals * torch.norm(rays_d[...,None,:], dim=-1), -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (torch.sum(weights, -1)  + 1e-6))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])
        # rgb_map = rgb_map + torch.cat([acc_map[..., None] * 0, acc_map[..., None] * 0, (1. - acc_map[..., None])], -1)

    return rgb_map, disp_map, acc_map, weights, depth_map