import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torchvision import models, utils

from arcface.iresnet import *


class fs_encoder_v2(nn.Module):
    def __init__(self, n_styles=18, opts=None, residual=False, use_coeff=False, resnet_layer=None, video_input=False, f_maps=512, stride=(1, 1)):
        super(fs_encoder_v2, self).__init__()

        self.n_styles = n_styles
        self.stride = stride
        self.opts = opts

        if not hasattr(opts, 'device'):
            opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("Loading ArcFace model from:", opts.arcface_model_path)
        arcface_state_dict = torch.load(opts.arcface_model_path, map_location='cpu')

        # Load ArcFace on CPU and extract layers
        full_model = iresnet50()
        full_model.load_state_dict(arcface_state_dict)
        full_model.eval()

        children = list(full_model.children())

        conv_layers = children[:3] if not video_input else [
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=False),
            *children[1:3]
        ]
        self.conv = nn.Sequential(*conv_layers)
        self.block_1 = children[3]
        self.block_2 = children[4]
        self.block_3 = children[5]
        self.block_4 = children[6]

        self.content_layer = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.PReLU(num_parameters=512),
            nn.Conv2d(512, 512, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(512),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.styles = None  # Will be initialized lazily in forward()

    def forward(self, x):
        device = x.device  # Get device from input tensor

        # Move all submodules to the correct device at runtime
        self.conv = self.conv.to(device)
        self.block_1 = self.block_1.to(device)
        self.block_2 = self.block_2.to(device)
        self.block_3 = self.block_3.to(device)
        self.block_4 = self.block_4.to(device)
        self.content_layer = self.content_layer.to(device)
        self.avg_pool = self.avg_pool.to(device)

        # Lazy initialization of style layers (saves GPU memory)
        if self.styles is None:
            self.styles = nn.ModuleList([
                nn.Linear(960 * 9, 512).to(device) for _ in range(self.n_styles)
            ])
            self.styles.to(device)

        latents = []
        features = []

        x = self.conv(x)
        x = self.block_1(x)
        features.append(self.avg_pool(x))

        x = self.block_2(x)
        features.append(self.avg_pool(x))

        x = self.block_3(x)
        content = self.content_layer(x)
        features.append(self.avg_pool(x))

        x = self.block_4(x)
        features.append(self.avg_pool(x))

        x = torch.cat(features, dim=1)
        x = x.view(x.size(0), -1)

        for layer in self.styles:
            latents.append(layer(x))

        out = torch.stack(latents, dim=1)
        return out, content
