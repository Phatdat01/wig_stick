import torch
from torch import nn
import gc

from models.Encoders import ClipBlendingModel, PostProcessModel
from models.Net import Net
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.save_utils import save_gen_image, save_latents


class Blending(nn.Module):
    """
    Module for transferring the desired hair color and post processing
    """

    def __init__(self, opts, net=None):
        super().__init__()
        self.opts = opts
        self.device = opts.device

        # Clear GPU cache and collect garbage to free memory before loading
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
            gc.collect()

        # Load main generator net to device
        self.net = net if net is not None else Net(self.opts)

        # Load blending encoder checkpoint on CPU
        blending_checkpoint = torch.load(self.opts.blending_checkpoint, map_location='cpu')
        self.blending_encoder = ClipBlendingModel(blending_checkpoint.get('clip', "ViT-B/32"))
        self.blending_encoder.load_state_dict(blending_checkpoint['model_state_dict'], strict=False)
        self.blending_encoder = self.blending_encoder.to(self.device).eval()
        if 'cuda' in str(self.device):
            self.blending_encoder = self.blending_encoder.half()

        # Load post-process model checkpoint on CPU, keep model on CPU for now
        self.post_process = PostProcessModel()
        pp_checkpoint = torch.load(self.opts.pp_checkpoint, map_location='cpu')
        self.post_process.load_state_dict(pp_checkpoint['model_state_dict'])
        self.post_process.eval()  # keep on CPU initially

        # Dilate/Erosion and downsampling modules
        self.dilate_erosion = DilateErosion(dilate_erosion=self.opts.smooth, device=self.device)
        self.downsample_256 = BicubicDownSample(factor=4)

    def _move_post_process_to_device(self):
        """
        Helper to move post_process model to device on demand.
        """
        current_device = next(self.post_process.parameters()).device
        if current_device != self.device:
            try:
                self.post_process = self.post_process.to(self.device)
                if 'cuda' in str(self.device):
                    self.post_process = self.post_process.half()
            except RuntimeError as e:
                print(f"Warning: Failed to move post_process to {self.device} due to OOM:\n{e}")
                print("Falling back to CPU for post_process.")
                self.post_process = self.post_process.to('cpu')

    def blend_images(self, align_shape, align_color, name_to_embed, **kwargs):
        with torch.no_grad():
            # Move post_process model to device just before inference if needed
            self._move_post_process_to_device()

            I_1 = name_to_embed['face']['image_norm_256'].to(self.device)
            I_2 = name_to_embed['shape']['image_norm_256'].to(self.device)
            I_3 = name_to_embed['color']['image_norm_256'].to(self.device)

            if 'cuda' in str(self.device):
                I_1, I_2, I_3 = I_1.half(), I_2.half(), I_3.half()

            mask_de = self.dilate_erosion.hair_from_mask(
                torch.cat([name_to_embed[x]['mask'].to(self.device) for x in ['face', 'color']], dim=0)
            )

            HM_1D, _ = mask_de[0][0].unsqueeze(0), mask_de[1][0].unsqueeze(0)
            HM_3D, HM_3E = mask_de[0][1].unsqueeze(0), mask_de[1][1].unsqueeze(0)

            latent_S_1 = name_to_embed['face']['S'].to(self.device)
            latent_S_3 = name_to_embed['color']['S'].to(self.device)
            latent_F_align = align_shape['latent_F_align'].to(self.device)
            HM_X = align_color['HM_X'].to(self.device)

            if 'cuda' in str(self.device):
                latent_S_1 = latent_S_1.half()
                latent_S_3 = latent_S_3.half()
                latent_F_align = latent_F_align.half()

            HM_XD, _ = self.dilate_erosion.mask(HM_X)
            target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)

            # Blending
            if I_1 is not I_3 or I_1 is not I_2:
                S_blend_6_18 = self.blending_encoder(
                    latent_S_1[:, 6:], latent_S_3[:, 6:], I_1 * target_mask, I_3 * HM_3E
                )
                S_blend = torch.cat((latent_S_1[:, :6], S_blend_6_18), dim=1)
            else:
                S_blend = latent_S_1

            I_blend, _ = self.net.generator(
                [S_blend], input_is_latent=True, return_latents=False,
                start_layer=4, end_layer=8, layer_in=latent_F_align
            )
            I_blend_256 = self.downsample_256(I_blend)

            # Free some memory before post processing
            del I_1, I_2, I_3, target_mask, HM_X, HM_3E
            torch.cuda.empty_cache()

            # Post Process
            S_final, F_final = self.post_process(S_blend, I_blend_256)
            I_final, _ = self.net.generator(
                [S_final], input_is_latent=True, return_latents=False,
                start_layer=5, end_layer=8, layer_in=F_final
            )

            if self.opts.save_all:
                exp_name = kwargs.get('exp_name', "")
                output_dir = self.opts.save_all_dir / exp_name
                save_gen_image(output_dir, 'Blending', 'blending.png', I_blend)
                save_latents(output_dir, 'Blending', 'blending.npz', S_blend=S_blend)

                save_gen_image(output_dir, 'Final', 'final.png', I_final)
                save_latents(output_dir, 'Final', 'final.npz', S_final=S_final, F_final=F_final)

            final_image = ((I_final[0] + 1) / 2).clamp(0, 1).float()
            return final_image.cpu()
