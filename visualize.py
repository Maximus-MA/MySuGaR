import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    AmbientLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    )
from pytorch3d.renderer.blending import BlendParams
from sugar_scene.gs_model import GaussianSplattingWrapper
from sugar_utils.spherical_harmonics import SH2RGB
import torchvision
from tqdm import tqdm


# setting
numGPU = 0
torch.cuda.set_device(numGPU)
source_path = os.path.expanduser('~/datasets/lego')
gs_checkpoint_path = './output/gs7000/lego'
iteration_to_load = 7000
load_gt_images = False
use_eval_split = False
n_skip_images_for_eval_split = 8


# Load Gaussian Splatting checkpoint 
print(f"\nLoading config {gs_checkpoint_path}...")

nerfmodel = GaussianSplattingWrapper(
    source_path=source_path,
    output_path=gs_checkpoint_path,
    iteration_to_load=iteration_to_load,
    load_gt_images=load_gt_images,
    eval_split=use_eval_split,
    eval_split_interval=n_skip_images_for_eval_split,
    )

print(f'{len(nerfmodel.training_cameras)} training images detected.')
print(f'The model has been trained for {iteration_to_load} steps.')
print(len(nerfmodel.gaussians._xyz) / 1e6, "M gaussians detected.")