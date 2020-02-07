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
        if cfg.IO and not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.cfg = cfg
        self.model = None
        self.use_gpu = False

        self.loss_func = None
        self.submodels = {}

        
    def save(self, epoch, solvers, acc):
        state = {}
        for solver in solvers:
            opt_state = solvers[solver].opt.state_dict()
            state[f"opt_{solver}"] = opt_state

        if isinstance(self.model, torch.nn.DataParallel): 
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
            
        state['model'] = model_state 
        if len(self.submodels) > 0:
            for submodel in enumerate(self.submodels):
                submodel_state = self.submodels[submodel].state_dict()
                state[f"submodel_{submodel}"] = submodel_state

        torch.save(state, os.path.join(self.save_path,'model_{:03}_{:.4f}.pth'.format(epoch, acc)))

    def load(self, path): 
        state = torch.load(path, map_location = torch.device('cpu'))
        model_state = self.model.state_dict()
        
        if 'model' in state:
            ckpt = state['model']
        else:
            ckpt = state

        self._load(ckpt, model_state)            
        self.model.load_state_dict(model_state)

        if len(self.submodels) > 0:
            for submodel in enumerate(self.submodels):
                submodel_state = self.submodels[submodel].state_dict()
                if f"submodel_{submodel}" in state:
                    ckpt = state[f"submodel_{submodel}"]
                    self._load(ckpt, submodel_state)
                    self.submodels[submodel].load_state_dict(submodel_state)

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
                logger.info(f"{m.name:<60}")
                if isinstance(inp, tuple):            
                    for i in inp:
                        size = i.size()
                        msg = f"{'input':>20}, "
                        for s in size:
                            msg += f"{s:>3} "
                        logger.info(msg)
                else:
                    size = inp.size()
                    msg = f"{'input':>20}, "
                    for s in size:
                        msg += f"{s:>3} "
                    logger.info(msg)
                size = out.size()
                msg = f"{'output':>20}, "
                for s in size:
                    msg += f"{s:>3} "
                logger.info(msg)
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

    @staticmethod
    def _load(src_state, trt_state):
        loaded_params = set()
        for layer, weight in src_state.items():
            if layer not in trt_state:
                logger.info("{}src_state {:55} ...... {}?{}".format(bcolors.RESET, layer, bcolors.WARNING, bcolors.RESET))
            else:
                src_w_shape = weight.size()
                trt_w_shape = trt_state[layer].size()
                if torch.isnan(weight).sum() == 0 and src_w_shape == trt_w_shape:
                    loaded_params.add(layer)
                    trt_state[layer] = weight
                    # logger.info("{}model {:55} ...... {}O{}".format(bcolors.RESET, layer, bcolors.OKGREEN, bcolors.RESET))
                else:
                    logger.info("{}model {:55} ...... {}X{}".format(bcolors.RESET, layer, bcolors.WARNING, bcolors.RESET))
                    logger.info(" => Shape (ckpt != model) {} != {}".format(src_w_shape, trt_w_shape))
        params = set(trt_state.keys())
        not_loaded_params = list(params.difference(loaded_params))
        for layer in not_loaded_params:
            logger.info("{}model {:55} ...... {}!{}".format(bcolors.RESET, layer, bcolors.WARNING, bcolors.RESET))  
        return trt_state