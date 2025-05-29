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

        # Ensure opts has device attribute
        if not hasattr(opts, 'device'):
            opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("Loading ArcFace model from:", opts.arcface_model_path)
        arcface_state_dict = torch.load(opts.arcface_model_path, map_location='cpu')

        # Instantiate and load weights
        resnet50 = iresnet50()
        resnet50.load_state_dict(arcface_state_dict)
        resnet50 = resnet50.to(opts.device)

        # Input conv layer
        if video_input:
            self.conv = nn.Sequential(
                nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                *list(resnet50.children())[1:3]
            )
        else:
            self.conv = nn.Sequential(*list(resnet50.children())[:3])

        # Define layers from the ArcFace model
        self.block_1 = list(resnet50.children())[3]  # 15–18
        self.block_2 = list(resnet50.children())[4]  # 10–14
        self.block_3 = list(resnet50.children())[5]  # 5–9
        self.block_4 = list(resnet50.children())[6]  # 1–4

        self.content_layer = nn.Sequential(
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.PReLU(num_parameters=512),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.styles = nn.ModuleList([
            nn.Linear(960 * 9, 512) for _ in range(n_styles)
        ])

    def forward(self, x):
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
