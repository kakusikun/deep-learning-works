import sys
import torch
import math
import numpy as np 
import src.solver.optimizers as opts
from src.solver.lr_schedulers import LRScheduler
import logging
logger = logging.getLogger("logger")

class Solver(): 
    def __init__(self, 
        cfg, 
        params_groups,
        lr=None,
        momentum=None,
        wd=None,
        lr_policy=None,
        opt_name=None):  
        self.lr = cfg.SOLVER.BASE_LR if lr is None else lr
        self.bias_lr_factor = cfg.SOLVER.BIAS_LR_FACTOR 
        self.momentum = cfg.SOLVER.MOMENTUM if momentum is None else momentum
        self.wd = cfg.SOLVER.WEIGHT_DECAY if wd is None else wd
        self.wd_factor = cfg.SOLVER.WEIGHT_DECAY_BIAS_FACTOR
        self.lr_policy = cfg.SOLVER.LR_POLICY if lr_policy is None else lr_policy
        self.opt_name = cfg.SOLVER.OPTIMIZER if opt_name is None else opt_name
        self.num_iter_per_epoch = cfg.SOLVER.ITERATIONS_PER_EPOCH

        # plateau
        self.gamma = cfg.SOLVER.GAMMA
        self.patience = cfg.SOLVER.PLATEAU_SIZE * self.num_iter_per_epoch 
        self.monitor_lr = 0.0
            
        self._model_analysis(params_groups, custom=cfg.SOLVER.CUSTOM)

        if self.opt_name == 'SGD':
            self.opt = torch.optim.SGD(self.params, momentum=self.momentum, nesterov=cfg.SOLVER.NESTEROV)
        elif self.opt_name == 'Adam':
            self.opt = torch.optim.Adam(self.params, amsgrad=cfg.SOLVER.AMSGRAD)
        elif self.opt_name == 'AdamW':
            self.opt = torch.optim.AdamW(self.params, amsgrad=cfg.SOLVER.AMSGRAD)
        elif self.opt_name == 'SGDW':
            self.opt = opts.SGDW(self.params, momentum=self.momentum, nesterov=cfg.SOLVER.NESTEROV)

        self.scheduler = LRScheduler(
            optimizer=self.opt,
            raw_policy=self.lr_policy,
            num_iter_per_epoch=self.num_iter_per_epoch,
        )
        
        if cfg.DB.USE_TRAIN:
            cfg.SOLVER.MAX_EPOCHS = max(list(self.scheduler.policy_schedule.keys()))

    def _model_analysis(self, params_groups, custom=[]):
        self.params = []
        # self.params = [{"params": params, "lr": self.lr, "weight_decay": self.wd}]
        num_params = 0.0
        for params in params_groups:
            for layer, p in params:
                #  try:
                if not p.requires_grad:
                    continue
                lr = self.lr
                wd = self.wd
                if "bias" in layer:
                    lr = self.lr * self.bias_lr_factor
                    wd = self.wd * self.wd_factor    
                for name, target, value in custom:
                    if name in layer:
                        if target == 'lr':
                            lr = value
                        elif target == 'wd':
                            wd = value
                        else:
                            logger.info("Unsupported optimizer parameter: {}".format(target))

                self.params += [{"params": p, "lr": lr, "weight_decay": wd}]
                num_params += p.numel()
        
        logger.info("Trainable parameters: {:.2f}M".format(num_params / 1000000.0))
    
    def lr_adjust(self, metrics, iters):
        if self.scheduler is not None:
            self.scheduler.step(metrics, iters)   
            self.monitor_lr = self.scheduler.monitor_lrs[0]
        else:
            self.monitor_lr = self.lr
    
    def zero_grad(self):        
        self.opt.zero_grad()

    def step(self):
        self.opt.step()





        

