from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

#from .networks.msra_resnet import get_pose_net
#from .networks.dlav0 import get_pose_net as get_dlav0
#from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.pose_dla_dcn import get_keypoint_net as get_ev200_backbone
#from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
#from .networks.large_hourglass import get_large_hourglass_net

_model_factory = {
  #'res': get_pose_net, # default Resnet with deconv
  #'dlav0': get_dlav0, # default DLAup
  #'dla': get_dla_dcn,
  #'resdcn': get_pose_net_dcn,
  #'hourglass': get_large_hourglass_net,
  'ev200': get_ev200_backbone,
}

def create_model(arch, heads, head_conv):
  '''
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  #print(num_layers)
  arch = arch[:arch.find('_')] if '_' in arch else arch
  #print(arch)
  get_model = _model_factory[arch]
  #print("I AM Here1")
  model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  '''
  #'''
  get_model = _model_factory[arch]
  model = get_model("/aiml/y00451616/project/DL/face/CenterNet_FaceDet/CenterNet_Berkeley_test/src/lib/wider_ev200_centernet.prototxt","/aiml/y00451616/project/DL/face/CenterNet_FaceDet/CenterNet_Berkeley_test/src/lib/rfcn_inception_for_centernet_20190923_iter_240000.caffemodel",heads=heads,head_conv=head_conv)
  #print("I AM HERE2")
  #'''
  return model

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  is_train = False
  if is_train:
    #for key, value in checkpoint.items():
    #  print(key)
    #print('########################################################')
    #print('########################################################')
    #for key, value in state_dict_.items():
    #  print(key,'corresponds to ',state_dict_[key].shape)
    state_dict = {}

    # convert data_parallal to model
    #####removed by WX Feb 4, 2020#################
    #for k in state_dict_:
    #  if k.startswith('module') and not k.startswith('module_list'):
    #    state_dict[k[7:]] = state_dict_[k]
    #  else:
    #    state_dict[k] = state_dict_[k]
    #####removed by WX Feb 4, 2020#################
    model_state_dict = model.state_dict()
    #for key, value in model_state_dict.items():
    #  print(key,'corresponds to ',model_state_dict[key].shape)
    # check loaded parameters and created model parameters
    #####removed by WX Feb 4, 2020#################
    #for k in state_dict:
    #  if k in model_state_dict:
    #    if state_dict[k].shape != model_state_dict[k].shape:
    #      print('Skip loading parameter {}, required shape{}, '\
    #            'loaded shape{}.'.format(
    #        k, model_state_dict[k].shape, state_dict[k].shape))
    #      state_dict[k] = model_state_dict[k]
    #  else:
    #    print('Drop parameter {}.'.format(k))
    #####removed by WX Feb 4, 2020##################
    #####removed by WX Feb 4, 2020##################
    #for k in model_state_dict:
    #  if not (k in state_dict):
    #    print('No param {}.'.format(k))
    #    state_dict[k] = model_state_dict[k]
    #####removed by WX Feb 4, 2020##################
    #model.load_state_dict(state_dict, strict=False)
    #####added by WX Feb 4, 2020####################
    for k in model_state_dict:
      k_name = k.split('.')
      if k_name[-1] == 'hor_conv_weight' or k_name[-1] == 'ver_conv_weight':
        continue    
      elif k_name[-1] == 'hor_conv_bias' or k_name[-1] == 'ver_conv_bias':
        continue
      elif k_name[-2] == 'bn':
        continue
      elif k_name[-1] == 'scale_weight' or k_name[-1] == 'scale_bias1':
        continue
      elif k_name[-1] == 'conv_weight':
        k_name_ori = ""
        for id1 in range(0,len(k_name)-2):
          k_name_ori = k_name_ori + k_name[id1] + "."
        state_dict[k]=state_dict_[k_name_ori+"weight"]
      elif k_name[-1] == 'bias' and len(k_name)>3:
        k_name_ori = ""
        for id1 in range(0,len(k_name)-2):
          k_name_ori = k_name_ori + k_name[id1] + "."
        #print(k_name_ori)
        #print(k_name)
        state_dict[k]=state_dict_[k_name_ori+"bias"]
      elif (k_name[-1] == "running_mean" or k_name[-1] == "running_var" or k_name[-1] == "weight") and len(k_name)>3:
        k_name_ori = ""
        for id1 in range(0,len(k_name)-2):
          k_name_ori = k_name_ori + k_name[id1] + "."
        state_dict[k]=state_dict_[k_name_ori+k_name[-1]][:(int(k_name[-2])+1)*4] 
      elif k_name[-1] == "bias1":
        k_name_ori = ""
        for id1 in range(0,len(k_name)-2):
          k_name_ori = k_name_ori + k_name[id1] + "."
        state_dict[k]=state_dict_[k_name_ori+"bias"][:(int(k_name[-2])+1)*4] 
      elif k_name[-1] == "num_batches_tracked":
        pass
      elif k_name[-1] == "deconv_weight":
        k_name_ori = ""
        for id1 in range(0,len(k_name)-1):
          k_name_ori = k_name_ori + k_name[id1] + "."
        state_dict[k]=state_dict_[k_name_ori+"weight"]
        #print(k_name)
        #os._exit(0)
      else:
        state_dict[k]=state_dict_[k]
  else:
    state_dict = state_dict_
  #####added by WX Feb 4, 2020####################
  model.load_state_dict(state_dict, strict=False)
  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

