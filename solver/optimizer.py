import torch
import math
import numpy as np
from solver.solvers import *
import glog

class Solver(): 
    def __init__(self, cfg, params):
        if cfg.OPTIMIZER.OPTIMIZER_NAME == 'SGD':
            self.opt = MySGD(params, 
                            lr       = cfg.OPTIMIZER.BASE_LR, 
                            momentum = cfg.OPTIMIZER.MOMENTUM, 
                            nesterov = cfg.OPTIMIZER.NESTEROV)
        elif cfg.OPTIMIZER.OPTIMIZER_NAME == 'Novo':
            self.opt = NovoGrad(params, amsgrad=False)
        
        self.findLR = cfg.OPTIMIZER.LR_RANGE_TEST        
        self.lr = cfg.OPTIMIZER.BASE_LR
        self.nwd = cfg.OPTIMIZER.WEIGHT_DECAY
        self.WD_NORMALIZED = cfg.OPTIMIZER.WD_NORMALIZED
        self.cycle_mult = cfg.OPTIMIZER.WARMRESTART_MULTIPLIER
        self.cycle_len = cfg.OPTIMIZER.WARMRESTART_PERIOD  
        self.num_iter_per_epoch = cfg.OPTIMIZER.ITERATIONS_PER_EPOCH
        self.annealing_mult = 1.0 
        self.start_iter = 0
        self.steps = 0
        self.epoch_to_num_cycles = 0
        self.lr_policy = cfg.OPTIMIZER.LR_POLICY
        self.lr_decay = cfg.OPTIMIZER.LR_DECAY
        self.cyclic_max_lr = cfg.OPTIMIZER.CYCLIC_MAX_LR
        self.cyclic_min_lr = self.cyclic_max_lr / 6.0
        self.max_epoch = cfg.OPTIMIZER.MAX_EPOCHS
        self.cum_epoch = self.cycle_len + 1

        if self.findLR:
            if self.num_iter_per_epoch > 1000:
                self.LRIncrement = pow((1.0/self.lr), (1.0/self.num_iter_per_epoch)) 
            else:
                factor = int(1000/self.num_iter_per_epoch)
                self.LRIncrement = pow((1.0/self.lr), (1.0/(self.num_iter_per_epoch * factor)))
    
    def _iter_start(self, cur_iter, cur_epoch):
        self.steps = cur_iter # + self.start_iter
        if self.findLR:
            self.annealing_mult = pow(self.LRIncrement, float(self.steps))
        else:
            if self.lr_policy == 'cosine':
                self.annealing_mult, self.epoch_to_num_cycles = self.cosineLR(float(self.steps), self.cycle_len, self.cycle_mult, self.num_iter_per_epoch)

                if cur_epoch == self.cum_epoch:
                    self.cum_epoch += self.epoch_to_num_cycles / self.num_iter_per_epoch
                    glog.info("Next lr restart is at {}th epoch".format(int(self.cum_epoch)))
                    if self.lr_decay :
                        lr_decay, _ = self.cosineLR(float(self.steps), self.max_epoch, 1, self.num_iter_per_epoch)
                        self.lr *= lr_decay
                        glog.info("Base lr decays, next decay at {}th epoch".format(int(self.cum_epoch)))

            if self.lr_policy == 'cyclic':
                self.cyclicalLR()      
        

        for param_group in self.opt.param_groups:
            param_group['lr'] = self.lr * self.annealing_mult
            if self.WD_NORMALIZED:
                param_group['weight_decay'] = (self.nwd / np.sqrt(self.num_iter_per_epoch * self.epoch_to_num_cycles)) * self.annealing_mult
            else:
                param_group['weight_decay'] = self.nwd
    
    def before_backward(self):        
        self.opt.zero_grad()

    def after_backward(self):
        self.opt.step()

    @staticmethod
    def cosineLR(batch_idx, cycle_len, cycle_mult, num_iter_per_epoch):      
               
        restart_period = cycle_len * num_iter_per_epoch

        while batch_idx/restart_period > 1.:
            batch_idx = batch_idx - restart_period
            restart_period = restart_period * cycle_mult

        radians = math.pi*(batch_idx/restart_period)

        return 0.5*(1.0 + math.cos(radians)), restart_period


    def cyclicalLR(self):
        # Scaler: we can adapt this if we do not want the triangular CLR
        scaler = lambda x: 1.

        # Additional function to see where on the cycle we are
        def relative(it, stepsize):
            cycle = math.floor(1 + it / (2 * stepsize))
            x = abs(it / stepsize - 2 * cycle + 1)
            return max(0, (1 - x)) * scaler(cycle)

        self.lr = self.cyclic_min_lr + (self.cyclic_max_lr - self.cyclic_min_lr) * relative(self.steps, 4 * self.num_iter_per_epoch)

        

