import os
import torch
import math
import torch.nn as nn
from collections import OrderedDict
import logging
logger = logging.getLogger("logger")
from tools import bcolors
from copy import deepcopy

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
        model_state = self.model.state_dict()
        loaded_params = set()
        if 'model' in state:
            ckpt = state['model']
        else:
            ckpt = state

        for layer, weight in ckpt.items():
            if layer not in model_state:
                logger.info("{}ckpt {:55} ...... {}?{}".format(bcolors.RESET, layer, bcolors.WARNING, bcolors.RESET))
            else:
                ckpt_w_shape = weight.size()
                model_w_shape = model_state[layer].size()
                if torch.isnan(weight).sum() == 0 and ckpt_w_shape == model_w_shape:
                    loaded_params.add(layer)
                    model_state[layer] = weight
                    # logger.info("{}model {:55} ...... {}O{}".format(bcolors.RESET, layer, bcolors.OKGREEN, bcolors.RESET))
                else:
                    logger.info("{}model {:55} ...... {}X{}".format(bcolors.RESET, layer, bcolors.WARNING, bcolors.RESET))
                    logger.info(" => Shape (ckpt != model) {} != {}".format(ckpt_w_shape, model_w_shape))
        params = set(model_state.keys())
        not_loaded_params = list(params.difference(loaded_params))
        for layer in not_loaded_params:
            logger.info("{}model {:55} ...... {}!{}".format(bcolors.RESET, layer, bcolors.WARNING, bcolors.RESET))    
            
        self.model.load_state_dict(model_state)

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

    def check_size(self, insize):
        def check_size_hooker(m, inp, out):
            if isinstance(out, torch.Tensor):
                logger.info("{:<60}".format(m.name))
                if isinstance(inp, tuple):            
                    for i in inp:
                        n,c,h,w = i.size()
                        logger.info("{:>20}{:>3} x {:>3} x {:>3} x {:>3}".format("input, ", n,c,h,w))
                else:
                    n,c,h,w = inp.size()
                    logger.info("{:>20}{:>3} x {:>3} x {:>3} x {:>3}".format("input, ", n,c,h,w))
                n,c,h,w = out.size()
                logger.info("{:>20}{:>3} x {:>3} x {:>3} x {:>3}".format("output, ", n,c,h,w))

        hooks = []
        for n, m in self.model.named_modules():
            m.name = n
            handle = m.register_forward_hook(check_size_hooker)
            hooks.append(handle)
        
        dummy_model = deepcopy(self.model)        
        weight = next(iter(dummy_model.parameters()))
        if weight.is_cuda:
            dummy_input = torch.rand(insize).cuda()
        else:
            dummy_input = torch.rand(insize)
        dummy_model.eval()
        with torch.no_grad():
            dummy_model(dummy_input)

        for hook in hooks:
            hook.remove()
