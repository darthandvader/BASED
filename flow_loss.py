from easydict import EasyDict as edict
from typing import Any, Dict, Tuple
import torch
import numpy as np
from nerf_utils import *
import torch.nn as nn
from batched_geometry_utils import *

def generate_adjacent_pair_list(n_views: int):
    """Generate list of possible exhaustive pairs, (Nx2). """
    pairs = []
    for i in range(0,n_views-1, 2):
        pairs.append([i, i+1])
        if i+10 < n_views:
            pairs.append([i+1,i+10])
        if i+5 < n_views:
            pairs.append([i+1,i+5])
    # pairs is N
    pairs = np.array(pairs)  # N x 2
    return torch.from_numpy(pairs.T)


def get_mask_valid_from_conf_map(p_r: torch.Tensor, corres_map: torch.Tensor, 
                                 min_confidence: float, max_confidence: float=None) -> torch.Tensor:
    channel_first = False
    if len(corres_map.shape) == 4:
        # (B, 2, H, W) or (B, H, W, 2)
        if corres_map.shape[1] == 2:
            corres_map = corres_map.permute(0, 2, 3, 1)
            channel_first = True
        if len(p_r.shape) == 3:
            p_r = p_r.unsqueeze(-1)
        if p_r.shape[1] == 1:
            p_r = p_r.permute(0, 2, 3, 1)
        h, w = corres_map.shape[1:3]
        valid_matches = corres_map[:, :, :, 0].ge(0) & corres_map[:, :, :, 0].le(w-1) & corres_map[:, :, :, 1].ge(0) & corres_map[:, :, :, 1].le(h-1)
        mask = p_r.ge(min_confidence)
        if max_confidence is not None:
            mask = mask & p_r.le(max_confidence)
        mask = mask & valid_matches.unsqueeze(-1)  # (B, H, W, 1)
        if channel_first:
            mask = mask.permute(0, 3, 1, 2)
    else:
        if corres_map.shape[0] == 2:
            corres_map = corres_map.permute(1, 2, 0)
        if len(p_r.shape) == 2:
            p_r = p_r.unsqueeze(-1)
            channel_first = True
        if p_r.shape[0] == 1:
            p_r = p_r.unsqueeze(1, 2, 0)
        h, w = corres_map.shape[:2]
        valid_matches = corres_map[:, :, 0].ge(0) & corres_map[:, :, 0].le(w-1) & corres_map[:, :, 1].ge(0) & corres_map[:, :, 1].le(h-1)
        mask = p_r.ge(min_confidence)
        if max_confidence is not None:
            mask = mask & p_r.le(max_confidence)
        mask = mask & valid_matches.unsqueeze(-1)  # (H, W, 1)
        if channel_first:
            mask = mask.permute(2, 0, 1)
    return mask 

class CorrespondenceBasedLoss:

    def __init__(self, H, W, focal, flow_net, images, poses, times, masks, nerf_net, device):
        default_cfg = edict({'matching_pair_generation': 'all', 
                             'min_nbr_matches': 500, 
                            
                             'pairing_angle_threshold': 30, 
                             'filter_corr_w_cc': False, 
                             'min_conf_valid_corr': 0.95, 
                             'min_conf_cc_valid_corr': 1./(1. + 1.5), 
                             })
        self.device = device
        self.train_images = images
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        self.H, self.W, self.B = H, W, images.shape[0]
        self.focal = focal
        self.grid = torch.stack((xx, yy), dim=-1).to(self.device).float()  # ( H, W, 2)
        self.grid_flat = self.grid[:, :, 1] * W + self.grid[:, :, 0]  # (H, W), corresponds to index in flattedned array (in H*W)
        self.grid_flat = self.grid_flat.to(self.device).long()
        self.train_poses = poses
        self.train_times = times
        self.train_masks = masks
        self.flow_net = flow_net
        self.nerf_net = nerf_net

        self.gt_corres_map_and_mask_all_to_all = None

        self.compute_correspondences(images, poses, masks)

    @torch.no_grad()
    def compute_correspondences(self, images, poses, masks):
        """Compute correspondences relating the input views. 

        Args:
            train_data (dataset): training dataset. The keys all is a dictionary, 
                                  containing the entire training data. 
                                  train_data.all has keys 'idx', 'image', 'intr', 'pose' 
                                  and all images of the scene already stacked here.

        """
        print('Computing flows')
        images = images.permute(0,3,1,2)
        H, W = images.shape[-2:]
        n_views = images.shape[0]

        combi_list = generate_adjacent_pair_list(n_views)

        print(f'Computing {combi_list.shape[1]} correspondence maps')

        corres_maps, conf_maps, flow_plot = self.flow_net.compute_flow_and_confidence_map_of_combi_list\
                (images, combi_list_tar_src=combi_list, plot=True, 
                use_homography=False)
        
        ## Shreya mask out pixels which fall in tool masks

        # confidence_map_masks = []
        # for src,trg in zip(combi_list[0],combi_list[1]):
        #     confidence_map_masks.append((self.train_masks[src]*self.train_masks[trg]).cpu().detach().numpy())

        # confidence_map_masks = torch.unsqueeze(torch.tensor(confidence_map_masks),1)
        conf_maps = conf_maps ##*confidence_map_masks

        ## Shreya code addition ends
        
        mask_valid_corr = get_mask_valid_from_conf_map(p_r=conf_maps.reshape(-1, 1, H, W), 
                                                       corres_map=corres_maps.reshape(-1, 2, H, W), 
                                                       min_confidence=0.95)
        

        self.corres_maps = corres_maps  # (combi_list.shape[1], 3, H, W)
        self.conf_maps = conf_maps
        self.mask_valid_corr = mask_valid_corr

        flow_pairs = (combi_list.cpu().numpy().T).tolist()  
        self.flow_pairs = flow_pairs
        # list of pairs, the target is the first element, source is second
        assert self.corres_maps.shape[0] == len(flow_pairs)

        filtered_flow_pairs = []
        for i in range(len(flow_pairs)):
            nbr_confident_regions = self.mask_valid_corr[i].sum()
            if nbr_confident_regions > 500:
                filtered_flow_pairs.append((i, flow_pairs[i][0], flow_pairs[i][1]))
                # corresponds to index_of_flow, index_of_target_image, index_of_source_image
        self.filtered_flow_pairs = filtered_flow_pairs
        print(f'{len(self.filtered_flow_pairs)} possible flow pairs')
        return
    
    def sample_valid_image_pair(self):
        """select an image pair in the filtered pair and retrieve corresponding 
        correspondence, confidence map and valid mask. 
        
        Returns: 
            if_self
            id_matching_view
            corres_map_self_to_other_ (H, W, 2)
            conf_map_self_to_other_ (H, W, 1)
            variance_self_to_other_ (H, W, 1) or None
            mask_correct_corr (H, W, 1)
        """
        id_in_flow_list = np.random.randint(len(self.filtered_flow_pairs))
        id_in_flow_tensor, id_self, id_matching_view = self.filtered_flow_pairs[id_in_flow_list]
        corres_map_self_to_other_ = self.corres_maps[id_in_flow_tensor].permute(1, 2, 0)[:, :, :2]  # (H, W, 2)
        conf_map_self_to_other_ = self.conf_maps[id_in_flow_tensor].permute(1, 2, 0)  # (H, W, 1)
        variance_self_to_other_ = None
        mask_correct_corr = self.mask_valid_corr[id_in_flow_tensor].permute(1, 2, 0)  # (H, W, 1)

        return id_self, id_matching_view, corres_map_self_to_other_, conf_map_self_to_other_, variance_self_to_other_, mask_correct_corr
    
    def render_image_at_specific_pose_and_rays():
        return
    
    def calculate_loss(self, rays_o, rays_d, coords, render_kwargs_train, i, poses, progress):
        self.train_poses = poses
        id_self, id_matching_view, corres_map_self_to_other_, conf_map_self_to_other_, variance_self_to_other_, mask_correct_corr = self.sample_valid_image_pair()
        B, H, W, focal = self.B, self.H, self.W, self.focal
        images = self.train_images.permute(0,2,3,1)
        poses_w2c = self.train_poses
        times = self.train_times

        pose_w2c_self = torch.eye(4).to(poses_w2c.device)
        pose_w2c_self[:3, :4] = poses_w2c[id_self] # the pose itself is just (3, 4)
        time_self = times[id_self].unsqueeze(0)
        pose_w2c_other = torch.eye(4).to(poses_w2c.device)
        pose_w2c_other[:3, :4] = poses_w2c[id_matching_view]  # (3, 4)
        time_other = times[id_matching_view].unsqueeze(0)

        corres_map_self_to_other = corres_map_self_to_other_.detach()
        conf_map_self_to_other = conf_map_self_to_other_.detach()
        mask_correct_corr = mask_correct_corr.detach().squeeze(-1)  # (H, W)
        corres_map_self_to_other_rounded = torch.round(corres_map_self_to_other).long()  # (H, W, 2)
        corres_map_self_to_other_rounded_flat = \
            corres_map_self_to_other_rounded[:, :, 1] * W + corres_map_self_to_other_rounded[:, :, 0] # corresponds to index in flattedned array (in H*W)
        
        pixels_in_self = self.grid[mask_correct_corr]  # [N_ray, 2], absolute pixel locations
        ray_in_self_int = self.grid_flat[mask_correct_corr]  # [N_ray]

        pixels_in_other = corres_map_self_to_other[mask_correct_corr] # [N_ray, 2], absolute pixel locations
        ray_in_other_int = corres_map_self_to_other_rounded_flat[mask_correct_corr]  # [N_ray]
        conf_values = conf_map_self_to_other[mask_correct_corr]  # [N_ray, 1] 

        if ray_in_self_int.shape[0] > 2048 // 2:
            random_values = torch.randperm(ray_in_self_int.shape[0],device=self.device)[:4096//2]
            ray_in_self_int = ray_in_self_int[random_values]
            pixels_in_self = pixels_in_self[random_values]

            pixels_in_other = pixels_in_other[random_values]
            ray_in_other_int = ray_in_other_int[random_values]
            conf_values = conf_values[random_values]

        
        pixels_in_self = pixels_in_self.type(torch.LongTensor)
        pixels_in_other = pixels_in_other.type(torch.LongTensor)
        
        rays_o_self = rays_o[pixels_in_self[:, 1], pixels_in_self[:, 0]]
        rays_o_other = rays_o[pixels_in_other[:, 1], pixels_in_other[:, 0]]

        rays_d_self = rays_d[pixels_in_self[:, 1], pixels_in_self[:, 0]]
        rays_d_other = rays_d[pixels_in_other[:, 1], pixels_in_other[:, 0]]

        batch_rays_self = torch.stack([rays_o_self, rays_d_self], 0)
        batch_rays_other = torch.stack([rays_o_other, rays_d_other], 0)

        rgb_self, disp_self, acc_self, extras_self = render(progress, H, W, focal, chunk=32768, rays=batch_rays_self, frame_time=time_self,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        depth_self = 1.0 / (disp_self + 1e-6)

        rgb_other, disp_other, acc_other, extras_other = render(progress, H, W, focal, chunk=32768, rays=batch_rays_other, frame_time=time_other,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        depth_other = 1.0 / (disp_other + 1e-6)

        intrinsic = torch.tensor([[focal,0,0],[0,focal,0],[0,0,1]]).to(device='cuda')

        cononical_self = get_caninical_3d_pts(pixels_in_self.to(device='cuda'), depth_self.to(device='cuda'), intrinsic, time_self, self.nerf_net, pose_w2c_self, progress)
        cononical_other = get_caninical_3d_pts(pixels_in_other.to(device='cuda'), depth_other.to(device='cuda'), intrinsic, time_other, self.nerf_net, pose_w2c_other, progress)
        diff = cononical_self - cononical_other
        
        delta = 1.
        loss = nn.functional.huber_loss(diff, torch.zeros_like(diff), reduction='none', delta=delta)

        return loss.sum() / (loss.nelement() + 1e-6)
    