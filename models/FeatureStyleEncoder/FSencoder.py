from argparse import Namespace
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import yaml
import sys

from PIL import Image
from tqdm import tqdm
from torchvision import transforms, utils

# Setup paths and environment
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_dir)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None

# Set training options
opts = Namespace(
    config='001',
    pretrained_model_path='pretrained_models/FeatureStyleEncoder/143_enc.pth',
    stylegan_model_path='pretrained_models/FeatureStyleEncoder/psp_ffhq_encode.pt',
    arcface_model_path='pretrained_models/FeatureStyleEncoder/backbone.pth',
    parsing_model_path='pretrained_models/FeatureStyleEncoder/79999_iter.pth',
    log_path='./logs/',
    resume=False,
    checkpoint='',
    checkpoint_noiser='',
    multigpu=False,
    input_path='./test/',
    save_path='./'
)

# Load configuration
config_path = os.path.join(current_dir, 'configs', f'{opts.config}.yaml')
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

from trainer import Trainer

def get_trainer(device='cuda'):
    # Set the desired device in opts for downstream modules
    opts.device = device

    # Initialize Trainer object with config and options
    trainer = Trainer(config, opts)

    # Initialize with paths to pretrained models
    trainer.initialize(
        stylegan_path=opts.stylegan_model_path,
        arcface_path=opts.arcface_model_path,
        parsing_path=opts.parsing_model_path
    )

    # Move entire trainer to the selected device
    trainer.to(device)

    # Load pretrained encoder weights (FeatureStyleEncoder)
    print(f"Loading encoder weights from {opts.pretrained_model_path}")
    enc_weights = torch.load(opts.pretrained_model_path, map_location=device)
    trainer.enc.load_state_dict(enc_weights)
    trainer.enc.eval()

    return trainer
