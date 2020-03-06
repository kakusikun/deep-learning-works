import os
import torch
import math
import torch.nn as nn
from collections import OrderedDict
import logging
logger = logging.getLogger("logger")
from tools import bcolors
from copy import deepcopy

class BaseGraph:
    def __init__(self, cfg):
        '''
        Customized the build function, need to specify
        1. model, a nn.Module
            - backbone
            - head
        2. loss_head, a function(feat, batch)
        3. sub_model (optional)
        '''
        self.cfg = cfg
        self.model = None
        self.loss_head = None
        self.use_gpu = False
        self.sub_models = {}
        
        self.set_save_path()
        self.build()

    def build(self):        
        raise NotImplementedError
        
    @staticmethod
    def save(path, model, sub_models=None, solvers=None, epoch=-1,  metric=-1):
        state = {}
        if solvers is not None:
            assert isinstance(solvers, dict)
            for solver in solvers:
                opt_state = solvers[solver].opt.state_dict()
                state[f"{solver}"] = opt_state

        if isinstance(model, torch.nn.DataParallel): 
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
            
        state['model'] = model_state 
        if sub_models is not None:
            assert isinstance(sub_models, dict)
            for sub_model in enumerate(sub_models):
                sub_model_state = sub_models[sub_model].state_dict()
                state[f"{sub_model}"] = sub_model_state

        torch.save(state, os.path.join(path,'model_{:03}_{:.4f}.pth'.format(epoch, metric)))

    def load(self, path, model=None, sub_models=None, solvers=None): 
        if not model:
            model = self.model
            
        state = torch.load(path, map_location = torch.device('cpu'))
               
        if 'model' in state:
            ckpt = state['model']            
        else:
            ckpt = state
        model_state = model.state_dict() 
        self._state_processing(ckpt, model_state)                    
        self.model.load_state_dict(model_state)

        if sub_models is not None:
            assert isinstance(sub_models, dict)
            for sub_model in enumerate(sub_models):
                if f"{sub_model}" in state:
                    sub_model_state = sub_models[sub_model].state_dict()
                    ckpt = state[f"{sub_model}"]
                    self._state_processing(ckpt, sub_model_state)
                    sub_models[sub_model].load_state_dict(sub_model_state)

        if solvers is not None:
            assert isinstance(solvers, dict)
            for solver in solvers:
                if f"{solver}" in state:
                    solvers[solver].opt.load_state_dict(state[f"{solver}"])

    def _initialize_weights(self):
        for m in self.model.modules():
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
    
    def to_gpu(self):
        gpu = os.environ['CUDA_VISIBLE_DEVICES']
        num_gpus = len(gpu.split(","))        
        if self.model is None:
            logger.info("Initial model first")
        else:
            if num_gpus > 0 and torch.cuda.device_count() > 0:
                logger.info("Use GPU")
                self.model = self.model.cuda()
                for sub_model in self.sub_models:
                    self.sub_models[sub_model] = self.sub_models[sub_model].cuda()
                self.use_gpu = True
            elif torch.cuda.device_count() == 0:
                logger.info("GPU is no found")
            else:
                logger.info("GPU is not used")

    def to_gpus(self):
        gpu = os.environ['CUDA_VISIBLE_DEVICES']
        num_gpus = len(gpu.split(","))

        if self.model is None:
            logger.info("Initial model first")
        else:            
            if num_gpus > 1 and torch.cuda.device_count() > 1:
                if self.use_gpu:
                    logger.info("Use GPUs: {}{}{}{}".format(bcolors.RESET, bcolors.OKGREEN, gpu, bcolors.RESET))
                    self.model = torch.nn.DataParallel(self.model)
                else:
                    logger.info("Use .cuda() first")
            else:
                logger.info("Use One GPU")

    def set_save_path(self):
        self.save_path = os.path.join(self.cfg.OUTPUT_DIR, 'weights')
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
    def _state_processing(src_state, trt_state):
        loaded_params = set()
        is_unknown = False
        is_not_fit = False
        for layer, weight in src_state.items():
            if layer not in trt_state:
                logger.info("{}src_state {:55} ...... {}?{}".format(bcolors.RESET, layer, bcolors.WARNING, bcolors.RESET))
                is_unknown = True
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
                    is_unknown = True
        params = set(trt_state.keys())
        not_loaded_params = list(params.difference(loaded_params))
        for layer in not_loaded_params:
            logger.info("{}model {:55} ...... {}!{}".format(bcolors.RESET, layer, bcolors.WARNING, bcolors.RESET))  
            is_not_fit = True
        
        if is_unknown:
            logger.info("Unknown Weights or Shape")
        else:
            if is_not_fit:
                logger.info("Unknown Layer in Model")
            else:
                logger.info("Model Loaded Successfully")
        
        