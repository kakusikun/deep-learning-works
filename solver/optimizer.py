import torch
import math
import numpy as np 
from solver.solvers import *
from solver.lr_schedulers import *
import logging
logger = logging.getLogger("logger")

class Solver(): 
    def __init__(self, cfg, params, _lr=None, _wd=None, _name=None, _lr_policy=None):   
        self.lr = cfg.SOLVER.BASE_LR if _lr is None else _lr
        self.monitor_lr = 0.0
        self.wd = cfg.SOLVER.WEIGHT_DECAY if _wd is None else _wd
        self.cycle_mult = cfg.SOLVER.WARMRESTART_MULTIPLIER
        self.cycle_len = cfg.SOLVER.WARMRESTART_PERIOD  
        self.num_iter_per_epoch = cfg.SOLVER.ITERATIONS_PER_EPOCH
        self.annealing_mult = 1.0
        self.lr_policy = cfg.SOLVER.LR_POLICY if _lr_policy is None else _lr_policy
        self.lr_steps = cfg.SOLVER.LR_STEPS
        self.opt_name = cfg.SOLVER.OPTIMIZER_NAME if _name is None else _name

        self.max_epoch = cfg.SOLVER.MAX_EPOCHS
        self.bias_lr_factor = cfg.SOLVER.BIAS_LR_FACTOR
        self.wd_factor = cfg.SOLVER.WEIGHT_DECAY_BIAS_FACTOR

        self.warmup = cfg.SOLVER.WARMUP
        self.gamma = cfg.SOLVER.GAMMA
        self.warmup_factor = cfg.SOLVER.WARMUP_FACTOR
        self.warmup_iters = cfg.SOLVER.WARMUP_SIZE * self.num_iter_per_epoch
        self.patience = cfg.SOLVER.PLATEAU_SIZE * self.num_iter_per_epoch 

        self._model_analysis(params)

        if self.opt_name == 'SGD':
            self.opt = torch.optim.SGD(self.params, momentum=cfg.SOLVER.MOMENTUM)
        elif self.opt_name == 'Adam':
            self.opt = torch.optim.Adam(self.params)
        
        if self.lr_policy == "plateau":
            self.scheduler = WarmupReduceLROnPlateau(optimizer=self.opt, 
                                                            mode="min",
                                                            gamma=self.gamma,
                                                            patience=self.patience,
                                                            warmup_factor=1.0/3,#self.warmup_factor,
                                                            warmup_iters=self.warmup_iters,
                                                            )
        else:
            self.scheduler = None
        
        logger.info("{} policy is used".format(self.lr_policy))


    def _model_analysis(self, params):
        self.params = []
        # self.params = [{"params": params, "lr": self.lr, "weight_decay": self.wd}]
        num_params = 0.0
        
        for layer, p in params:
            try:
                if not p.requires_grad:
                    continue
                lr = self.lr
                wd = self.wd
                if "bias" in layer:
                    lr = self.lr * self.bias_lr_factor
                    wd = self.wd * self.wd_factor                
                self.params += [{"params": p, "lr": lr, "weight_decay": wd}]
                num_params += p.numel()
            except:
                logger.info("{:50} ...... skipped".format(layer))
        
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



        

