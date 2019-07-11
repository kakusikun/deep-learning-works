import os
import sys
import torch
import math
import torch.nn as nn
from collections import OrderedDict
from model.OSNet import OSNet
from model.ResNet import ResNet34
from model.ResNet import ResNet18
from model.RMNet import RMNet
from model.utility import FC, CenterLoss, AMSoftmax, CrossEntropyLossLS
from model.model_manager import TrainingManager
import glog

# class RMNetManager(TrainingManager):
#     def __init__(self, cfg):
#         super(RMNetManager, self).__init__(cfg)        

#         if cfg.TASK == "imagenet":
#             self.models = OrderedDict({"main": RMNet(b=[4,8,10,11], cifar10=False, reid=False)})
#             self.loss_funcs = OrderedDict({"CE": nn.CrossEntropyLoss()})
#         elif cfg.TASK == "cifar10":
#             self.models = OrderedDict({"main": RMNet(b=[4,8,10,11], cifar10=True, reid=False)})
#             self.loss_funcs = OrderedDict({"CE": nn.CrossEntropyLoss()})
#         elif cfg.TASK == "reid":
#             self.models = OrderedDict({"main": RMNet(b=[4,8,10,11], cifar10=False, reid=True)})
#             self.loss_funcs = OrderedDict({"center_loss": CenterLoss(256, cfg.MODEL.NUM_CLASSES), 
#                                         "amsoftmax": AMSoftmax(256, cfg.MODEL.NUM_CLASSES),
#                                         "push_loss": })    
#         else:
#             glog.info("Task {} is not supported".format(cfg.TASK))  
#             sys.exit(1)

#         self._check_model()

#     def _initialize_weights(self):
#         for name in self.models.keys():            
#             for m in self.models[name].modules():
#                 if isinstance(m, nn.Conv2d):
#                     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                     m.weight.data.normal_(0, math.sqrt(2. / n))
#                     if m.bias is not None:
#                         m.bias.data.zero_()
#                     #  nn.init.kaiming_normal_(m.weight, a = 0, mode = 'fan_in')
#                     #  if m.bias:
#                         #  nn.init.constant_(m.bias, 0.0)
#                 elif isinstance(m, nn.BatchNorm2d):
#                     nn.init.constant_(m.weight, 1.0)
#                     nn.init.constant_(m.bias, 0.0)
#                 elif isinstance(m, nn.Linear):
#                     if name == "id_feat":
#                         nn.init.normal_(m.weight, std = 0.001)
#                         if m.bias:
#                             nn.init.constant_(m.bias, 0.0)
#                     else:
#                         nn.init.kaiming_normal_(m.weight, a = 0, mode = 'fan_out')
#                         nn.init.constant_(m.bias, 0.0)
                        
