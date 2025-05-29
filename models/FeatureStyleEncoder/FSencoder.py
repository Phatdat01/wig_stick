from argparse import Namespace
import os
import sys
import torch
import yaml
from PIL import Image
import gc

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_dir)

from trainer import Trainer

# Improve performance and compatibility
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None

# Define model options
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

# Load config
config = yaml.load(open(os.path.join(current_dir, 'configs', f'{opts.config}.yaml'), 'r'), Loader=yaml.FullLoader)

def get_trainer(device='cuda'):
    try:
        # Use CPU if CUDA is not available
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available. Switching to CPU.")
            device = 'cpu'

        opts.device = device

        # Clean memory
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

        # Initialize trainer
        trainer = Trainer(config, opts)
        trainer.initialize(
            opts.stylegan_model_path,
            opts.arcface_model_path,
            opts.parsing_model_path
        )
        trainer.to(device)

        # Load encoder weights
        print(f"Loading encoder weights from {opts.pretrained_model_path}")
        state_dict = torch.load(opts.pretrained_model_path, map_location=device)

        # Filter out keys not in the encoder
        encoder_state_dict = {
            k: v for k, v in state_dict.items() if k in trainer.enc.state_dict()
        }
        trainer.enc.load_state_dict(encoder_state_dict, strict=False)
        trainer.enc.eval()

        return trainer

    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory. Switching to CPU.")
        return get_trainer(device='cpu')
