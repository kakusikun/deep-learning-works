from bisect import bisect_right
import torch

import math
from torch._six import inf
from collections import Counter
from functools import partial

from torch.optim import Optimizer
import logging
logger = logging.getLogger("logger")

# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=10,
        warmup_method="linear",
        last_epoch=-1, 
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

class WarmupReduceLROnPlateau(object):
    def __init__(self, optimizer, mode='min', gamma=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='abs',
                 cooldown=0, min_lr=0, eps=1e-8,
                 warmup_factor=1.0 / 3,
                 warmup_iters=500,
                 last_iter=0):

        if gamma >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.gamma = gamma

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_iters = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_iter = last_iter
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lrs = list(map(lambda group: group['initial_lr'], self.optimizer.param_groups))
        self.monitor_lrs = self.base_lrs

        logger.info("Plateau lr policy is used")

    def _reset(self):
        """Resets num_bad_iters counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_iters = 0

    def step(self, metrics=None, iters=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        if metrics is None:
            raise ValueError

        current = float(metrics)
        if iters is None:
            iters = self.last_iter + 1
        self.last_iter = iters

        
        if self.last_iter <= self.warmup_iters:
            alpha = self.last_iter / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha

            self._warmup_lr([base_lr * warmup_factor for base_lr in self.base_lrs])
        else:
            if self.is_better(current, self.best):
                self.best = current
                # logger.info('Iteration {:5d}: Find Best:{:.4f} after {} iters'.format(iters, self.best, self.num_bad_iters))
                self.num_bad_iters = 0
            else:
                self.num_bad_iters += 1
                # if self.num_bad_iters % 100 == 0:            
                #     logger.info('Iteration {:5d}: has been {} stagnants. Best:{:.4f}'.format(iters, self.num_bad_iters, self.best))

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_iters = 0  # ignore any bad epochs in cooldown

            if self.num_bad_iters > self.patience:
                self._reduce_lr(iters)
                self.cooldown_counter = self.cooldown
                self.num_bad_iters = 0

    def _reduce_lr(self, iters):
        self.monitor_lrs = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.gamma, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr                
            self.monitor_lrs.append(new_lr)
        logger.info('Iteration {:5d}: reducing learning rate to {:.4e} with {} stagnant.'.format(iters, new_lr, self.num_bad_iters))

    def _warmup_lr(self, lrs):
        self.monitor_lrs = []
        for lr, param_group in zip(lrs, self.optimizer.param_groups):
            param_group['lr'] = lr
            self.monitor_lrs.append(lr)

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key not in {'optimizer', 'is_better'}}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)


class WarmupCosineLR():
    def __init__(
        self,
        optimizer,
        num_iter_per_epoch,
        warmup_factor=1.0 / 3,
        warmup_iters=10,
        anneal_mult=2,
        anneal_period=10,
    ):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        self.num_iter_per_epoch = num_iter_per_epoch
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.anneal_mult = anneal_mult
        self.anneal_period = anneal_period
        self.last_iter = 0

        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lrs = list(map(lambda group: group['initial_lr'], self.optimizer.param_groups))
        self.monitor_lrs = []

        logger.info("Cosine lr policy is used")

    def step(self, metrics=None, iters=None):
        if iters is None:
            iters = self.last_iter + 1
        self.last_iter = iters

        if self.last_iter < self.warmup_iters:
            alpha = self.last_iter / self.warmup_iters
            factor = self.warmup_factor * (1 - alpha) + alpha
        else:
            factor = self.cosineLR(self.last_iter - self.warmup_iters, self.anneal_period, self.anneal_mult, self.num_iter_per_epoch)
        self._adjust_lr(factor)

    def _adjust_lr(self, factor):
        self.monitor_lrs = []
        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            param_group['lr'] = base_lr * factor
            self.monitor_lrs.append(base_lr * factor)

    
    @staticmethod
    def cosineLR(batch_idx, T_0, T_mult, num_iter_per_epoch):      
               
        restart_period = T_0 * num_iter_per_epoch

        while batch_idx/restart_period > 1.:
            batch_idx = batch_idx - restart_period
            restart_period = restart_period * T_mult

        radians = math.pi*(batch_idx/restart_period)

        return 0.5*(1.0 + math.cos(radians))
