# **[BASED](https://sites.google.com/ucsd.edu/based-arclab-2023/)**

Implementation for **[BASED: Bundle-Adjusting Surgical Endoscopic Dynamic Video Reconstruction using Neural Radiance Fields](https://arxiv.org/abs/2309.15329)** by Shreya Saha, Sainan Liu, Shan Lin, Jingpei Lu, Michael Yip.

## Dataset - 

Please email ssaha@ucsd.edu to get access to the formatted dataset.

## External Submodule Dependency - 

```
git submodule update --init --recursive
git submodule update --recursive --remote
```

## Part 1 - Training pose matrix and Radiance Fields Simultaneously

In this part, we train the learnable pose matrix and the NeRF models at the same time for 200 iterations. We do not use GT or reference depths in this step. So do not use the flag use_depth here.

Train without using flow correspondence loss.
```
python3 nerf_pose_training.py --config configs/rectified18_1.txt --gt_fgmask
```

Train using flow correspondence loss.
```
python3 nerf_pose_training_with_flow_loss.py --config configs/rectified18_1.txt --gt_fgmask
```

## Part 2 - Freeze pose matrix and only train Radiance Fields

In this part, we use the fixed weights of the pose matrix, and only train the NeRF model

```
python3 nerf_training_after_pose_freezing.py --config configs/rectified18_1.txt --i_weights 100 --gt_fgmask --use_depth
```

You can opt not to use the mask and depth flags in order to view results from the ablation study section. 

## Evaluation (Inference Results) 

Use the flag --test_time_optim only if you want the model to optimise poses during test time. The results in the paper were based on test time optimised poses. 

```
python3 nerf_eval.py --config configs/rectified18_1.txt --test_time_optim
```

In case you want to add novel view renderings, use the mask --novel_views. The other results will be disabled on using this mask. A spiral zoom will be created for each test view. 

```
python3 nerf_eval.py --config configs/rectified18_1.txt --test_time_optim --novel_views
```

## Evaluation Metrics

In case there are no ground truth masks - 

```
python3 eval_rgb_without_masks.py --gt_dir logs/rectified09/nerf_onlytest_time_107000/gt --img_dir logs/rectified09/nerf_onlytest_time_107000/estim/
```

In case there are ground truth masks - 

```
python3 eval_rgb.py --gt_dir logs/rectified09/nerf_onlytest_time_107000/gt --img_dir logs/rectified09/nerf_onlytest_time_107000/estim/ --mask_dir logs/rectified09/nerf_onlytest_time_107000/gt_masks/
```

For further details, please contact Shreya Saha (ssaha@ucsd.edu)



