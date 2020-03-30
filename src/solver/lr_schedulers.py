from bisect import bisect_right
from torch._six import inf
from collections import Counter
from functools import partial

import torch
from collections import OrderedDict
import math
from torch.optim import Optimizer
import logging
logger = logging.getLogger("logger")

class StepLR():
    def get_factor(self, schedule, iters):
        factor = schedule['gamma'] ** schedule['power']
        return factor

class CosineLR():
    def __init__(self, num_iter_per_epoch):
        self.num_iter_per_epoch = num_iter_per_epoch
    
    def get_factor(self, schedule, iters, last_epoch):
        factor, n_cycle = self.cosineLR(
            iters - last_epoch * self.num_iter_per_epoch, 
            schedule['T_0'], 
            schedule['T_MULT'], 
            self.num_iter_per_epoch
        ) 
        factor *= schedule['decay'] ** n_cycle
        return factor

    @staticmethod
    def cosineLR(batch_idx, T_0, T_mult, num_iter_per_epoch):      
        n_cycle = 1
        restart_period = T_0 * num_iter_per_epoch

        while batch_idx/restart_period > 1.:
            batch_idx = batch_idx - restart_period
            restart_period = restart_period * T_mult
            n_cycle += 1

        radians = math.pi*(batch_idx/restart_period)

        return 0.5*(1.0 + math.cos(radians)), n_cycle

class WarmLR():
    def __init__(self, num_iter_per_epoch):
        self.num_iter_per_epoch = num_iter_per_epoch
    
    def get_factor(self, iters, target_epoch, last_epoch):
        factor = (iters - last_epoch * self.num_iter_per_epoch) / ((target_epoch - last_epoch) * self.num_iter_per_epoch)
        return factor

class PlateauLR():
    def __init__(self, num_iter_per_epoch):
        self.num_iter_per_epoch = num_iter_per_epoch
        self.n_drop = 0
        self._init_is_better()
        self._reset()

    def get_factor(self, schedule, metrics, iters):
        if metrics is None:
            raise ValueError
        current = float(metrics)
        
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_iters = 0
        else:
            self.num_bad_iters += 1

        if self.num_bad_iters > schedule['plateau_size'] * self.num_iter_per_epoch:
            logger.info(f"Reducing learning rate with {self.num_bad_iters} stagnant")
            self.n_drop += 1            
            self.num_bad_iters = 0        

        factor = schedule['gamma'] ** self.n_drop
        return factor

    def _init_is_better(self, mode='min', threshold=1e-4, threshold_mode='abs'):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)
        
    def _reset(self):
        """Resets num_bad_iters counter and cooldown counter."""
        self.best = self.mode_worse
        self.num_bad_iters = 0

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


class LRScheduler():
    '''
    Args:
        raw_policy (str):
            a string that indicates the scheduler and when to use
            supported format:
                none, do nothing
                warm-A, 
                    warmup with number of epochs of A
                cosine-A-B-C-D, 
                    with number of cycles of A 
                    and lr decays with factor B after each cycle
                    cosine annealing with T_0 C 
                    and T_MULT D
                step-A-B-C, 
                    step the lr with number of epochs of A 
                    with power B of C
                plateau-A-B-C,
                    until epoch A
                    step the lr with factor B
                    if there is no update for minimum of metric lasting for C epochs

            
    '''
    def __init__(self,
        optimizer,
        raw_policy,
        num_iter_per_epoch,
    ):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.policy_schedule = self._policy_parser(raw_policy)
        self.num_iter_per_epoch = num_iter_per_epoch
        self.permanent_factor = 1.0

        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lrs = list(map(lambda group: group['initial_lr'], self.optimizer.param_groups))

        self.warmlr = WarmLR(num_iter_per_epoch)
        self.cosinelr = CosineLR(num_iter_per_epoch)
        self.steplr = StepLR()
        self.plateaulr = PlateauLR(num_iter_per_epoch)

    def step(self, metrics=None, iters=None):
        last_epoch = 0        
        for e in self.policy_schedule:
            if iters <= e * self.num_iter_per_epoch:
                target_epoch = e
                break
            target_epoch = e
            last_epoch = e
        curr_schedule = self.policy_schedule[target_epoch]
        
        if curr_schedule['policy'] == 'warm':
            factor = self.warmlr.get_factor(iters, target_epoch, last_epoch)
        elif curr_schedule['policy'] == 'cosine':
            factor = self.cosinelr.get_factor(curr_schedule, iters, last_epoch)
        elif curr_schedule['policy'] == 'step':
            factor = self.steplr.get_factor(curr_schedule, iters)
            self.permanent_factor = factor
        elif curr_schedule['policy'] == 'plateau':
            factor = self.plateaulr.get_factor(curr_schedule, metrics, iters)
        elif curr_schedule['policy'] == 'none':
            factor = 1.0 * self.permanent_factor
        
        self._adjust_lr(factor)

    def _adjust_lr(self, factor):
        self.monitor_lrs = []
        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            param_group['lr'] = base_lr * factor
            self.monitor_lrs.append(base_lr * factor)

    def _policy_parser(self, raw_policy):
        policy = OrderedDict()
        _policy = raw_policy.split(" ")
        target_epoch = 0
        for p in _policy:
            parsed = p.split("-")
            if parsed[0] == 'warm':
                duration = int(parsed[1])  
                policy[target_epoch+duration] = {'policy':'warm'}
            elif parsed[0] == 'cosine':
                duration = self.calc_cos_epochs(int(parsed[4]), int(parsed[3]), int(parsed[1]))
                policy[target_epoch+duration] = {'policy':'cosine', 'T_0': int(parsed[3]), 'T_MULT': int(parsed[4]), 'decay': float(parsed[2])}
            elif parsed[0] == 'step':
                duration = int(parsed[1])
                policy[target_epoch+duration] = {'policy':'step', 'gamma': float(parsed[3]), 'power': float(parsed[2])}
            elif parsed[0] == 'plateau':
                duration = int(parsed[1])
                policy[target_epoch+duration] = {'policy':'plateau', 'gamma': float(parsed[2]), 'plateau_size': float(parsed[3])}
            elif parsed[0] == 'none':
                duration = 0
                policy[target_epoch] = {'policy':'none'}
            else:
                raise TypeError
            target_epoch += duration
        return policy

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

    @staticmethod
    def calc_cos_epochs(base, period, cycles):
        n_epochs = 0
        for i in range(cycles):
            n_epochs += period * (base ** i)
        return n_epochs
    




if __name__ == "__main__":
    from src.model.backbone.hacnn import hacnn
    import torch
    model = hacnn()
    opt = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.1)
    scheduler = LRScheduler(
        opt,
        # "warm-2 cosine-3-0.9-2-2 step-4-0.1-0 step-4-0.1-1",
        "warm-2 cosine-3-0.9-2-2",
        1
    )

    for i in range(1,100):
        scheduler.step(iters=i)