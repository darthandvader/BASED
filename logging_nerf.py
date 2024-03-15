import os
import imageio
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from apex import amp
except ImportError:
    pass


def save_nerf_model(args, basedir, expname, nerf_dir,i, global_step, render_kwargs_train, optimizer, depth_maps):
    path = os.path.join(basedir, expname, nerf_dir, '{:06d}.tar'.format(i))
    save_dict = {
        'global_step': global_step,
        'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'ray_importance_maps': ray_importance_maps.cpu().numpy(),
        'depth_maps': depth_maps.cpu().numpy() if depth_maps is not None else None
    }
    if render_kwargs_train['network_fine'] is not None:
        save_dict['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict()

    if args.do_half_precision:
        save_dict['amp'] = amp.state_dict()
    torch.save(save_dict, path)
    print('Saved checkpoints at', path)

