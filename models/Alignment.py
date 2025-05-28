import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn

from models.CtrlHair.shape_branch.config import cfg as cfg_mask
from models.CtrlHair.shape_branch.solver import get_hair_face_code, get_new_shape, Solver as SolverMask
from models.Encoders import RotateModel
from models.Net import Net, get_segmentation
from models.sean_codes.models.pix2pix_model import Pix2PixModel, SEAN_OPT, encode_sean, decode_sean
from utils.image_utils import DilateErosion
from utils.save_utils import save_vis_mask, save_gen_image, save_latents


class Alignment(nn.Module):
    def __init__(self, opts, latent_encoder=None, net=None):
        super().__init__()
        self.opts = opts
        self.latent_encoder = latent_encoder
        self.net = net if net else Net(self.opts)

        torch.cuda.empty_cache()
        self.sean_model = Pix2PixModel(SEAN_OPT).to(opts.device)
        self.sean_model.eval()

        solver_mask = SolverMask(cfg_mask, device='cpu', local_rank=-1, training=False)
        solver_mask.gen.load_state_dict(torch.load('pretrained_models/ShapeAdaptor/mask_generator.pth', map_location='cpu'))
        self.mask_generator = solver_mask.gen.to(opts.device)
        torch.cuda.empty_cache()

        checkpoint = torch.load(self.opts.rotate_checkpoint, map_location='cpu')
        self.rotate_model = RotateModel()
        self.rotate_model.load_state_dict(checkpoint['model_state_dict'])
        self.rotate_model.to(self.opts.device).eval()
        torch.cuda.empty_cache()

        self.dilate_erosion = DilateErosion(dilate_erosion=self.opts.smooth, device=self.opts.device)
        self.to_bisenet = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def shape_module(self, im_name1, im_name2, name_to_embed, only_target=True, **kwargs):
        device = self.opts.device
        img1_in = name_to_embed[im_name1]['image_256'].to(device)
        img2_in = name_to_embed[im_name2]['image_256'].to(device)
        latent_W_1 = name_to_embed[im_name1]["W"].to(device)
        latent_W_2 = name_to_embed[im_name2]["W"].to(device)
        inp_mask1 = name_to_embed[im_name1]['mask'].to(device)
        inp_mask2 = name_to_embed[im_name2]['mask'].to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                same_image = torch.equal(img1_in, img2_in)
                if not same_image:
                    rotate_to = self.rotate_model(latent_W_2[:, :6], latent_W_1[:, :6])
                    rotate_to = torch.cat((rotate_to, latent_W_2[:, 6:]), dim=1)
                    I_rot, _ = self.net.generator([rotate_to], input_is_latent=True, return_latents=False)
                    I_rot_to_seg = ((I_rot + 1) / 2).clamp(0, 1)
                    I_rot_to_seg = self.to_bisenet(I_rot_to_seg)
                    rot_mask = get_segmentation(I_rot_to_seg)
                else:
                    I_rot = None
                    rot_mask = inp_mask2

                if not same_image:
                    face_1, hair_1 = get_hair_face_code(self.mask_generator, inp_mask1[0, 0])
                    face_2, hair_2 = get_hair_face_code(self.mask_generator, rot_mask[0, 0])
                    target_mask = get_new_shape(self.mask_generator, face_1, hair_2)[None, None].to(device)
                else:
                    target_mask = inp_mask1

                hair_mask_target = (target_mask == 13).float().to(device)

        if self.opts.save_all:
            exp_name = kwargs.get('exp_name', "")
            output_dir = self.opts.save_all_dir / exp_name
            if I_rot is not None:
                save_gen_image(output_dir, 'Shape', f'{im_name2}_rotate_to_{im_name1}.png', I_rot)
            save_vis_mask(output_dir, 'Shape', f'mask_{im_name1}.png', inp_mask1)
            save_vis_mask(output_dir, 'Shape', f'mask_{im_name2}.png', inp_mask2)
            save_vis_mask(output_dir, 'Shape', f'mask_{im_name2}_rotate_to_{im_name1}.png', rot_mask)
            save_vis_mask(output_dir, 'Shape', f'mask_{im_name1}_{im_name2}_target.png', target_mask)

        if only_target:
            return {'HM_X': hair_mask_target}
        else:
            hair_mask1 = (inp_mask1 == 13).float()
            hair_mask2 = (inp_mas