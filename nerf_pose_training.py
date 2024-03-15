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

    if args.llffhold>0 and args.dataset != 'Hamlyn':
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

def train():
    parser = config_parser()
    args = parser.parse_args()

    print("use_depth :: ", args.ref_depths)

    times, render_times, poses, render_poses, i_train, i_test, hwf, near, far, depth_maps, images, masks = load_dataset(args)

    min_time, max_time = times[i_train[0]], times[i_train[-1]]
    assert min_time >= 0., "time must start at 0"
    assert max_time <= 1., "max time must be 1"

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    split_ind = i_train
    if args.render_test:
        split_ind = i_test
    render_poses = np.array(poses[split_ind])
    render_times = np.array(times[split_ind])
    images = images[split_ind]
    if masks is not None:
        masks = masks[split_ind]
    train_poses = poses[i_train]
    poses = poses[split_ind]
    if depth_maps is not None:
        depth_maps = depth_maps[split_ind]
    times = times[split_ind]
    
    basedir = args.basedir
    expname = args.expname
    nerf_dir = "nerf_and_pose"
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    os.makedirs(os.path.join(basedir, expname, nerf_dir), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    print('Log directory:', os.path.join(basedir, expname))

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, nerf_model_extras = create_nerf(args, nerf_dir, args.lrate_nerf_simul)
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
    
    N_rand = args.N_rand
    use_batching = not args.no_batching

    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    times = torch.Tensor(times).to(device)
    if masks is not None:
        masks = torch.Tensor(masks).to(device)
        if nerf_model_extras['ray_importance_maps'] is None:
            ray_importance_maps = ray_sampling_importance_from_masks(masks)
        else:
            ray_importance_maps = torch.Tensor(nerf_model_extras['ray_importance_maps']).to(device)
    # print("hems",ray_importance_maps.shape)
    if depth_maps is not None:
        if nerf_model_extras['depth_maps'] is None:
            depth_maps = torch.Tensor(depth_maps).to(device)
        else:
            depth_maps = torch.Tensor(nerf_model_extras['depth_maps']).to(device)

    
    # if use_batching:
    #     rays_rgb = torch.Tensor(rays_rgb).to(device)

    print('images shape', images.shape)
    print('poses shape', poses.shape)
    print('times shape', times.shape)

    if masks is not None:
        print('masks shape', masks.shape)
    if depth_maps is not None:
        print('depth shape', depth_maps.shape)
        print('close depth:', close_depth, 'inf depth:', inf_depth)

    N_iters = args.N_iter + 1
    print('Begin')

    writer = SummaryWriter(os.path.join(basedir, expname, 'summaries'))

    pose_net = FirstTwoColunmnsPoseParameters(nbr_poses=poses.shape[0], 
                                                           initial_poses_w2c=poses, 
                                                           device='cuda')
    ckpts = [os.path.join(args.basedir, args.expname, nerf_dir, f) for f in sorted(os.listdir(os.path.join(args.basedir, args.expname, nerf_dir))) if '.pth' in f and 'pose' in f]
    start_pose = 0
    if len(ckpts) != 0:
        pose_net.load_state_dict(torch.load(ckpts[-1]))
        print("pose net uploaded from checkpoint :: ", ckpts[-1])
        # start_pose = int(ckpts[-1][33:-4])
    optimizer1 = getattr(torch.optim,'Adam')
    optimizer_pose = optimizer1([dict(params=pose_net.parameters(), lr=3e-4)])
    scheduler = getattr(torch.optim.lr_scheduler,'ExponentialLR')
    lr_pose = 3e-3
    max_iter = N_iters
    lr_pose_end = 1.e-7 
    gamma = (lr_pose_end/lr_pose)**(1./max_iter)
    kwargs = {'gamma':gamma }
    scheduler_pose = scheduler(optimizer_pose,**kwargs)
    N_iters = poses.shape[0]*100*2 + 1
    start = start + 1
    img_i = -1
    for i in trange(start, N_iters):
        torch.cuda.empty_cache()

        img_i = img_i+1
        img_i = img_i%poses.shape[0]
        
        target = images[img_i]
        poses = pose_net.get_w2c_poses() 
        pose = poses[img_i, :3, :4]
        frame_time = times[img_i]

        if masks is not None:
            mask = masks[img_i]
            ray_importance_map = ray_importance_maps[img_i]
        if depth_maps is not None:
            depth_map = depth_maps[img_i]

        if N_rand is not None:
            rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            # print("ray_o", rays_o)
            # print("rays_d", rays_d)
            if i < args.precrop_iters:
                dH = int(H//2 * args.precrop_frac)
                dW = int(W//2 * args.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                        torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                    ), -1)
                if i == start:
                    print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            # print("masks",ray_importance_map.shape) ## NOT NONE 
            if masks is None or args.no_mask_raycast:
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            elif masks is not None:
                tmp = ray_importance_map[coords[:, 0].long(), coords[:, 1].long()]
                # print("tmp",tmp.shape)
                select_inds, _, cdf = importance_sampling_coords(ray_importance_map[coords[:, 0].long(), coords[:, 1].long()].unsqueeze(0), N_rand)
                select_inds = torch.max(torch.zeros_like(select_inds), select_inds)
                select_inds = torch.min((coords.shape[0] - 1) * torch.ones_like(select_inds), select_inds)
                select_inds = select_inds.squeeze(0)
                # print("confirming shape ::", select_inds.shape, N_rand)
                
            # print("select_inds",select_inds.shape,N_rand)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            if depth_maps is not None:
                depth_s = depth_map[select_coords[:, 0], select_coords[:, 1]]
                if not args.no_ndc:
                    depth_s = depth_s / ((inf_depth - close_depth) + 1e-6)

                # Apply depth-guided ray sampling ==> Shreya yep use GT depth to do depth guided ray sampling
                if not args.no_depth_sampling:
                    bds_dict = {
                        'near' : depth_s.detach().clone() + 1e-6,
                        'far' : args.depth_sampling_sigma,
                    }
                    render_kwargs_train.update(bds_dict)
            if masks is not None and args.mask_loss:
                mask_s = mask[select_coords[:, 0], select_coords[:, 1]]
                mask_s = mask_s.unsqueeze(-1)
            else:
                mask_s = None

        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, frame_time=frame_time,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        
        
        if mask_s is not None:
            rgb = rgb * mask_s
            target_s = target_s * mask_s
        
        img_loss = img2mse(rgb, target_s)
        psnr = mse2psnr(img_loss)
        loss = img_loss
        if depth_maps is not None:
            if args.depth_loss_weight > 1e-16:
                pred_depth = 1.0 / (disp + 1e-6)
                if mask_s is not None:
                    pred_depth = pred_depth * mask_s
                    depth_s = depth_s * mask_s
                depth_loss = F.huber_loss(pred_depth, depth_s, delta=0.2)
                loss = loss + args.depth_loss_weight * depth_loss
            else:
                depth_loss = torch.Tensor([-1.0]).to(device)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        if args.do_half_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        if i%poses.shape[0] == 0:
            optimizer.step()
            optimizer_pose.step()
            optimizer.zero_grad()
            optimizer_pose.zero_grad()
            scheduler_pose.step()
        # print("learning rate :: ", scheduler_pose.get_lr())
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        
        refinement_round = i // args.depth_refine_period
        if not args.no_depth_refine and depth_maps is not None and i % args.depth_refine_period == 0 and refinement_round <= args.depth_refine_rounds:
            print('Render RGB and depth maps for refinement...')
            
            refinement_save_path = os.path.join(basedir, expname, 'refinement{:04d}'.format(refinement_round))
            if not os.path.exists(refinement_save_path):
                os.makedirs(refinement_save_path)
            depth_prev_save_path = os.path.join(refinement_save_path, 'depth_prev')
            depth_refined_save_path = os.path.join(refinement_save_path, 'depth_refined')
            if not os.path.exists(depth_prev_save_path):
                os.makedirs(depth_prev_save_path)
            if not os.path.exists(depth_refined_save_path):
                os.makedirs(depth_refined_save_path)

            with torch.no_grad():
                rgbs_t, disps_t = render_path_gpu(poses, times, hwf, args.chunk, render_kwargs_test)

                masks_gt = masks # [N_train, H, W]

                # Refine depth maps
                depth_t = (1.0 / (disps_t + 1e-6)) * (inf_depth - close_depth)
                depth_gt = depth_maps

                max_depth = depth_maps.max()

                ## Shreya - comenting this out for now, uncomment if you want to see the depth maps
                # for j in i_train:
                #     imageio.imwrite(os.path.join(depth_prev_save_path, 'depth_{:0d}.png'.format(j)), to8b((depth_maps[j] / max_depth).cpu().numpy()))

                depth_diff = torch.pow(depth_t - depth_gt, 2) * masks_gt # [N_train, H, W]
                depth_diff = depth_diff.reshape(depth_diff.shape[0], -1) # [N_train, H x W]
                quantile = torch.quantile(depth_diff, 1.0 - args.depth_refine_quantile, dim=1, keepdim=True) # [N_train, 1]
                depth_to_refine = (depth_diff > quantile).reshape(*depth_t.shape) # [N_train, H, W]
                depth_gt[depth_to_refine] = depth_t[depth_to_refine]
                depth_maps = depth_gt

                max_depth = depth_maps.max()

                ## Shreya - comenting this out for now, uncomment if you want to see the depth maps
                # for j in i_train:
                #     imageio.imwrite(os.path.join(depth_refined_save_path, 'depth_{:0d}.png'.format(j)), to8b((depth_maps[j] / max_depth).cpu().numpy()))

                save_dict = {
                    'rounds': refinement_round,
                    'quantile': quantile.cpu().numpy(),
                    'depth_diff': depth_diff.cpu().numpy(),
                    'depth_to_refine': depth_to_refine.cpu().numpy()
                }
                torch.save(save_dict, os.path.join(refinement_save_path, 'depth_refine_info.tar'))

                del disps_t, depth_t, depth_gt, depth_to_refine, depth_diff, quantile

                del rgbs_t, masks_gt

                print('\nRefinement finished, intermediate results saved at', refinement_save_path)


        if i%args.i_weights==0:
            save_nerf_model(args, basedir, expname, nerf_dir,i, global_step, render_kwargs_train, optimizer, depth_maps)
            pose_path = os.path.join(basedir, expname, nerf_dir, 'pose_{:06d}.pth'.format(i))
            torch.save(pose_net.state_dict(), pose_path)

        if i % 100 == 0:
            tqdm_txt = f"[TRAIN] Iter: {i} Img Loss: {img_loss.item()} PSNR: {psnr.item()}"
            if args.add_tv_loss:
                tqdm_txt += f" TV: {tv_loss.item()}"
            if depth_maps is not None:
                tqdm_txt += f" Depth Loss: {depth_loss.item()}"
            tqdm.write(tqdm_txt)

            writer.add_scalar('simul_photo_loss', img_loss.item(), i)
            writer.add_scalar('simul_psnr', psnr.item(), i)
            if 'rgb0' in extras:
                writer.add_scalar('simul_loss0', img_loss0.item(), i)
                writer.add_scalar('simul_psnr0', psnr0.item(), i)
            if args.add_tv_loss:
                writer.add_scalar('simul_tv', tv_loss.item(), i)
            if depth_maps is not None:
                writer.add_scalar('simul_depth', depth_loss.item(), i)
            plotting_poses = plot_training_poses(poses, i)
            writer.add_image('poses_learnt', plotting_poses['training poses'], i) 
        
        
        
        del loss, img_loss, psnr, target_s
        if 'rgb0' in extras:
            del img_loss0, psnr0
        if args.add_tv_loss:
            del tv_loss
        if depth_maps is not None:
            del depth_loss, depth_s
        del rgb, disp, acc, extras

        global_step += 1

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()