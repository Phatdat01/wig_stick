"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_network import BaseNetwork
from .loss import *
# from sean_codes.models.networks import *
from . import *
from .encoder import *
import util.util as util


def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = 'sean_codes.models.networks.' + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network

    return network


def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()

    netG_cls = find_network_using_name(opt.netG, 'generator')
    parser = netG_cls.modify_commandline_options(parser, is_train)
    if is_train:
        netD_cls = find_network_using_name(opt.netD, 'discriminator')
        parser = netD_cls.modify_commandline_options(parser, is_train)
    netE_cls = find_network_using_name('conv', 'encoder')
    parser = netE_cls.modify_commandline_options(parser, is_train)

    return parser


def create_network(cls, opt):
    net = cls(opt)
    net.print_network()
    
    # Check if CUDA is available and if we have GPU IDs to use
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        net.cuda()
        print(f"CUDA is available. Using GPU {opt.gpu_ids}")
    else:
        print("CUDA is not available or no GPU specified. Falling back to CPU.")
    
    # Initialize weights
    net.init_weights(opt.init_type, opt.init_variance)
    return net



def define_G(opt):
    netG_cls = find_network_using_name(opt.netG, 'generator')
    return create_network(netG_cls, opt)


def define_D(opt):
    netD_cls = find_network_using_name(opt.netD, 'discriminator')
    return create_network(netD_cls, opt)


def define_E(opt):
    # there exists only one encoder type
    netE_cls = find_network_using_name('conv', 'encoder')
    return create_network(netE_cls, opt)
