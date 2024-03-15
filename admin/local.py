import os

class EnvironmentSettings:
    def __init__(self, data_root='', debug=False):
        self.workspace_dir = 'workspace_dir'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = 'tensorboard_dir'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir    # Directory for saving other models pre-trained networks
        self.eval_dir = 'eval_dir'    # Base directory for saving the evaluations. 
        self.log_dir = 'log_dir'
        self.pretrained_defs = 'pre_trained_models'
        self.llff = 'data/llff'
        self.hamlyn = 'data/Hamlyn/image_depth_data'
        self.endonerf = 'data/endonerf_sample_datasets'
        self.dtu = ''
        self.dtu_depth = ''
        self.dtu_mask = ''
        self.replica = ''