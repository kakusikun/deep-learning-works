import os
import torch
import math
import torch.nn as nn
from collections import OrderedDict
from model.OSNet import OSNet
from model.ResNet import ResNet34
from model.ResNet_cifar10 import ResNet18
from model.RMNet import RMNet
from model.utility import FC, CenterPushLoss, AMCrossEntropyLossLSR, CenterPushTupletLoss
import glog

class ModelManager():
    def __init__(self, cfg):
        self.savePath = os.path.join(cfg.OUTPUT_DIR, "weights")

        # self.models = OrderedDict({"main": OSNet(r=[64,96,128], b=[2,2,2], cifar10=True), "fc":FC(512, cfg.MODEL.NUM_CLASSES)})
        # self.models = OrderedDict({"main": RMNet(b=[4,8,10,11], cifar10=False, reid=False), "fc":FC(256, cfg.MODEL.NUM_CLASSES)})
        self.models = OrderedDict({"main": RMNet(b=[4,8,10,11], cifar10=False, reid=True, trick=True),
                                #   "local_loss": CenterPushTupletLoss(256, cfg.MODEL.NUM_CLASSES),
                                  "glob_loss": AMCrossEntropyLossLSR(256, cfg.MODEL.NUM_CLASSES, m=0.0, s=1.0, eps=0.1)})
        #  self.models = OrderedDict({"main": RMNet(b=[4,8,10,11], cifar10=False, reid=True),
                                   #  "glob_loss": AMCrossEntropyLossLSR(256, cfg.MODEL.NUM_CLASSES)})
        
        self.models = OrderedDict({"main": ResNet18(), "FC": FC(512, cfg.MODEL.NUM_CLASSES, False)})

        self.params = []
        num_params = 0
        for name in self.models.keys():
            self.params.append({"params": self.models[name].parameters()})
            num_params += sum([p.numel() for p in self.models[name].parameters() if p.requires_grad])
        
        glog.info("Total trainable parameters: {:.2f}M".format(num_params / 1000000.0))

        if cfg.RESUME:
            glog.info("Resuming model from {}".format(cfg.RESUME))
            self.loadPath = cfg.RESUME
            self.load_model()
        elif cfg.EVALUATE:
            glog.info("Evaluating model from {}".format(cfg.EVALUATE))
            self.loadPath = cfg.EVALUATE
            self.load_model()
        # else:
        #     self._initialize_weights()
        
        
    def save_model(self, epoch, optimizer, accuracy):
        
        if not os.path.exists(self.savePath):
            os.mkdir(self.savePath)

        for key in self.models.keys():
            state = {}
            if key == "main":
                opt_state = optimizer.opt.state_dict()
                state['optimizer'] = opt_state

            if isinstance(self.models[key], torch.nn.DataParallel):
                model_state = self.models[key].module.state_dict()
            else:
                model_state = self.models[key].state_dict()
            
            state['model_state_dict'] = model_state   
            torch.save(state, os.path.join(self.savePath,'model_{}_{}_{:.4f}.pth'.format(key, epoch, accuracy)))

    def load_model(self): 
        
        for i in range(0, int(len(self.loadPath)/2)):
            state = torch.load(self.loadPath[2*i+1])
            if 'model_state_dict' in state or 'model' in state:
                try:
                    checkpoint = state['model_state_dict']
                except:
                    checkpoint = state['model']

                model_state = self.models[self.loadPath[2*i]].state_dict()

                checkpointRefine = {k: v for k, v in checkpoint.items() if k in model_state and torch.isnan(v).sum() == 0}

                model_state.update(checkpointRefine)

                self.models[self.loadPath[2*i]].load_state_dict(model_state)
            else:
                checkpoint = state

                model_state = self.models[self.loadPath[2*i]].state_dict()

                checkpointRefine = {k: v for k, v in checkpoint.items() if k in model_state and torch.isnan(v).sum() == 0}

                model_state.update(checkpointRefine)

                self.models[self.loadPath[2*i]].load_state_dict(model_state)
        return self.models

    def _initialize_weights(self):
        for name in self.models.keys():            
            for m in self.models[name].modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
