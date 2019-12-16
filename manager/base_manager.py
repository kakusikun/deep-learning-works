import os
import torch
import math
import torch.nn as nn
from collections import OrderedDict
import logging
logger = logging.getLogger("logger")
from tools import bcolors

class BaseManager():
    def __init__(self, cfg):
        self.save_path = os.path.join(cfg.OUTPUT_DIR, "weights")
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.cfg = cfg
        self.model = None
        self.use_gpu = False

        self.loss_func = None
        self.submodels = {}

    def _check_model(self):
        if self.cfg.EVALUATE:
            logger.info("Evaluating model from {}".format(self.cfg.EVALUATE))
            self.loadPath = self.cfg.EVALUATE
            self.load_model()
        elif self.cfg.RESUME:
            logger.info("Resuming model from {}".format(self.cfg.RESUME))
            self.loadPath = self.cfg.RESUME
            self.load_model()     
        
    def save_model(self, epoch, solvers, acc):
        state = {}
        for solver in solvers:
            opt_name = "opt_{}".format(solver)
            opt_state = solvers[solver].opt.state_dict()
            state[opt_name] = opt_state

        if isinstance(self.model, torch.nn.DataParallel): 
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
            
        state['model'] = model_state 
        if len(self.submodels) > 0:
            for submodel in enumerate(self.submodels):
                submodel_name = "submodel_{}".format(submodel)
                submodel_state = self.submodels[submodel].state_dict()
                state[submodel_name] = submodel_state

        torch.save(state, os.path.join(self.save_path,'model_{:03}_{:.4f}.pth'.format(epoch, acc)))

    def load_model(self): 
        state = torch.load(self.loadPath, map_location = torch.device('cpu'))
        loaded_weights = {}
        if len(state.keys()) < 10:
            for key in state.keys():            
                if 'model' in key:
                    checkpoint = state[key]
                    model_state = self.model.state_dict()
                    
                    for k, _ in model_state.items():
                        loaded_weights[k] = False

                    checkpointRefine = {}             
                    for k, v in checkpoint.items():
                        if k in model_state and torch.isnan(v).sum() == 0:
                            ckpt_w_shape = v.size()
                            model_w_shape = model_state[k].size()
                            if ckpt_w_shape == model_w_shape:
                                checkpointRefine[k] = v
                                loaded_weights[k] = True
                                logger.info("{}model {:55} ...... {}loaded{}".format(bcolors.RESET, k, bcolors.OKGREEN, bcolors.RESET))
                            else:
                                logger.info("{}model {:55} ...... {}not loaded{}".format(bcolors.RESET, k, bcolors.WARNING, bcolors.RESET))
                                logger.info(" => Shape (ckpt != model) {} != {}".format(ckpt_w_shape, model_w_shape))
                        else:
                            logger.info("{}state {:55} ...... {}skipped{}".format(bcolors.RESET, k, bcolors.WARNING, bcolors.RESET))
                    
                    for k in loaded_weights.keys():
                        if not loaded_weights[k]:
                            logger.info("{}model {:55} ...... {}not loaded{}".format(bcolors.RESET, k, bcolors.WARNING, bcolors.RESET))     
                        
                    model_state.update(checkpointRefine)
                    self.model.load_state_dict(model_state)
                    
                #  elif 'loss' in key:
                    #  idx = int(key.split("_")[-1])
                    #  self.loss_has_param[idx].load_state_dict(state[key])
                else:
                    logger.info("{} is skipped".format(key))
        else:
            checkpoint = state
            model_state = self.model.backbone.state_dict()

            for k, _ in model_state.items():
                loaded_weights[k] = False

            checkpointRefine = {}             
            for k, v in checkpoint.items():
                if k in model_state and torch.isnan(v).sum() == 0:
                    checkpointRefine[k] = v
                    loaded_weights[k] = True
                    logger.info("{}{:55} ...... {}loaded{}".format(bcolors.RESET, k, bcolors.OKGREEN, bcolors.RESET))
                else:
                    logger.info("{}{:55} ...... {}skipped{}".format(bcolors.RESET, k, bcolors.WARNING, bcolors.RESET))

            for k in loaded_weights.keys():
                if not loaded_weights[k]:
                    logger.info("{}{:55} ...... {}not loaded{}".format(bcolors.RESET, k, bcolors.WARNING, bcolors.RESET))     

            model_state.update(checkpointRefine)
            self.model.backbone.load_state_dict(model_state)
        logger.info(bcolors.RESET)

    def _initialize_weights(self):
        raise NotImplementedError
    
    def use_multigpu(self):
        gpu = os.environ['CUDA_VISIBLE_DEVICES']
        num_gpus = len(gpu.split(","))
        logger.info("Using GPU: {}{}{}{}".format(bcolors.RESET, bcolors.OKGREEN, gpu, bcolors.RESET))

        if self.model is not None and torch.cuda.device_count() >= num_gpus and num_gpus > 0:
            self.use_gpu = True
            if num_gpus > 1: 
                logger.info("Use Multi-GPUs")
                self.model = torch.nn.DataParallel(self.model).cuda()
            else:
                logger.info("Use Single-GPU")
                self.model = self.model.cuda()
        else:
            if self.model is None:
                logger.info("Initial model first")
            elif torch.cuda.is_available():
                logger.info("GPU is no found")
            else:
                logger.info("GPU is not used")

    def set_save_path(self, path):
        self.save_path = os.path.join(self.cfg.OUTPUT_DIR, path)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
