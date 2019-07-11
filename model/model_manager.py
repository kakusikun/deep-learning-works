import os
import torch
import math
import torch.nn as nn
from collections import OrderedDict
import glog

class TrainingManager():
    def __init__(self, cfg):
        self.savePath = os.path.join(cfg.OUTPUT_DIR, "weights")

        self.cfg = cfg
        self.model = None

        self.loss_func = None
        self.loss_has_param = []

    def _check_model(self):
        if self.cfg.RESUME:
            glog.info("Resuming model from {}".format(self.cfg.RESUME))
            self.loadPath = self.cfg.RESUME
            self.load_model()
        elif self.cfg.EVALUATE:
            glog.info("Evaluating model from {}".format(self.cfg.EVALUATE))
            self.loadPath = self.cfg.EVALUATE
            self.load_model()
        else:
           self._initialize_weights()        
        
    def save_model(self, epoch, opts, acc):
        
        if not os.path.exists(self.savePath):
            os.mkdir(self.savePath)

        state = {}
        for i, opt in enumerate(opts):
            opt_name = "opt_{}".format(i)
            opt_state = opt.state_dict()
            state[opt_name] = opt_state

        if isinstance(self.model, torch.nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
            
        state['model'] = model_state 
        if len(self.loss_has_param) > 0:
            for i, loss in enumerate(self.loss_has_param):
                loss_name = "loss_{}".format(i)
                loss_state = loss.state_dict()
                state[loss_name] = loss_state

        torch.save(state, os.path.join(self.savePath,'model_{:03}_{:.4f}.pth'.format(epoch, acc)))

    def load_model(self): 
        state = torch.load(self.loadPath)
        for key in state.keys():            
            if 'model' in key:
                checkpoint = state[key]
                if self.cfg.MODEL.PRETRAIN == "imagenet":
                    model_state = self.model.backbone.state_dict()
                else:
                    model_state = self.model.state_dict()

                checkpointRefine = {}             
                for k, v in checkpoint.items():
                    if k in model_state and torch.isnan(v).sum() == 0:
                        checkpointRefine[k] = v
                        glog.info("{:60} ...... loaded".format(k))
                    else:
                        glog.info("{:60} ...... skipped".format(k))
                    
                model_state.update(checkpointRefine)
                if self.cfg.MODEL.PRETRAIN == "imagenet":
                    self.model.backbone.load_state_dict(model_state)
                else:
                    self.model.load_state_dict(model_state)
                
            elif 'loss' in key:
                idx = int(key.split("_")[-1])
                self.loss_has_param[idx].load_state_dict(state[key])
            else:
                glog.info("{} is skipped".format(key))
            
    def _initialize_weights(self):
        raise NotImplementedError
                        
