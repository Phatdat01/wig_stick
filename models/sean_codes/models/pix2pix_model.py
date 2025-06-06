"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
from argparse import Namespace
from glob import glob

import numpy as np
import torch

from models.sean_codes.util import util
from . import networks


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                # fake_image, _ = self.generate_fake(input_semantics, real_image)
                obj_dic = data['path']
                fake_image = self.save_style_codes(input_semantics, real_image, obj_dic)
            return fake_image
        elif mode == 'UI_mode':
            with torch.no_grad():
                # fake_image, _ = self.generate_fake(input_semantics, real_image)

                ################### some problems here
                obj_dic = data['obj_dic']
                # if isinstance(obj_dic, str):
                #     obj_dic = [obj_dic]
                fake_image = self.use_style_codes(input_semantics, real_image, obj_dic)
            return fake_image
        elif mode == 'style_code':
            with torch.no_grad():
                style_codes = self.netG.Zencoder(input=real_image, segmap=input_semantics)
            return style_codes
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.use_gpu():
            for param in ['label', 'instance', 'image']:
                if param in data and data[param] is not None:
                    data[param] = data[param].to('cpu', non_blocking=True)  # Ensure CPU usage

            if 'obj_dic' in data:
                for idx in range(19):
                    if data['obj_dic'][str(idx)]['ACE'].device.type == 'cpu':
                        data['obj_dic'][str(idx)]['ACE'] = data['obj_dic'][str(idx)]['ACE'].cpu()
        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()

        input_semantics = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        return input_semantics, data['image']

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}

        fake_image = self.generate_fake(
            input_semantics, real_image, compute_kld_loss=self.opt.use_vae)

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last color_texture is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer color_texture
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                              * self.opt.lambda_vgg

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):

        fake_image = self.netG(input_semantics, real_image)

        return fake_image

    ###############################################################

    def save_style_codes(self, input_semantics, real_image, obj_dic):

        fake_image = self.netG(input_semantics, real_image, obj_dic=obj_dic)

        return fake_image

    def use_style_codes(self, input_semantics, real_image, obj_dic):
        fake_image = self.netG(input_semantics, real_image, obj_dic=obj_dic)

        return fake_image

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0


def load_average_feature():
    ############### load average features
    # average_style_code_folder = 'styles_test/mean_style_code/mean/'
    average_style_code_folder = 'models/sean_codes/styles_test/mean_style_code/median/'
    input_style_dic = {}

    ############### hard coding for categories
    for i in range(19):
        input_style_dic[str(i)] = {}
        average_category_folder_list = glob(os.path.join(average_style_code_folder, str(i), '*.npy'))
        average_category_list = [os.path.splitext(os.path.basename(name))[0] for name in
                                 average_category_folder_list]

        for style_code_path in average_category_list:
            input_style_dic[str(i)][style_code_path] = torch.from_numpy(
        np.load(os.path.join(average_style_code_folder, str(i), style_code_path + '.npy')))
    return input_style_dic


def change_status(model, new_status):
    for m in model.modules():
        if hasattr(m, 'status'):
            m.status = new_status


def encode_sean(sean_model, images, labels):
    data = {'label': labels,
            'instance': torch.tensor(0),
            'image': images,
            'path': ['temp/temp_npy']}
    change_status(sean_model, 'test')
    img_codes = sean_model(data, mode='style_code')  # [2, 19, 512]
    return img_codes


def decode_sean(sean_model, image_code, target_mask):
    obj_dic = load_average_feature()

    for idx in range(19):
        cur_code = image_code[0, idx]
        if not torch.all(cur_code == 0):
            obj_dic[str(idx)]['ACE'] = cur_code

    temp_face_image = torch.zeros((0, 3, 256, 256))  # place holder

    data = {'label': target_mask,
            'instance': torch.tensor(0),
            'image': temp_face_image.clone().detach(),
            'obj_dic': obj_dic}
    change_status(sean_model, 'UI_mode')
    generated = sean_model(data, mode='UI_mode')[0]  # [3, 256, 256]
    return generated


SEAN_OPT = Namespace(name='CelebA-HQ_pretrained', gpu_ids=[0], checkpoints_dir='pretrained_models/sean_checkpoints',
                     model='pix2pix', norm_G='spectralspadesyncbatch3x3', norm_D='spectralinstance',
                     norm_E='spectralinstance', phase='test', batchSize=1, preprocess_mode='resize_and_crop',
                     load_size=256, crop_size=256, aspect_ratio=1.0, label_nc=19, contain_dontcare_label=False,
                     output_nc=3, dataroot='./datasets/cityscapes/', dataset_mode='custom', serial_batches=True,
                     no_flip=True, nThreads=28, max_dataset_size=9223372036854775807, load_from_opt_file=False,
                     cache_filelist_write=False, cache_filelist_read=False, display_winsize=256, netG='spade', ngf=64,
                     init_type='xavier', init_variance=0.02, z_dim=256, no_instance=True, nef=16, use_vae=False,
                     results_dir='./results/', which_epoch='latest', how_many=float("inf"), status='test', config='001',
                     gpu='0', need_crop=True, no_blending=False, num_upsampling_layers='normal', no_pairing_check=False,
                     label_dir='datasets/CelebA-HQ/test/labels', image_dir='datasets/CelebA-HQ/test/images',
                     instance_dir='', isTrain=False, semantic_nc=19)
