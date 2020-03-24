import torch
from collections import OrderedDict
import math
from torch.optim import Optimizer
import logging
logger = logging.getLogger("logger")

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
        self.policy_schedule = self.policy_parser(raw_policy)
        self.num_iter_per_epoch = num_iter_per_epoch
        self.last_iter = 0
        self.permanent_factor = 1.0

        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lrs = list(map(lambda group: group['initial_lr'], self.optimizer.param_groups))
        self.monitor_lrs = []

        logger.info("Cosine lr policy is used")

    def step(self, metrics=None, iters=None):
        if iters is None:
            iters = self.last_iter + 1
        self.last_iter = iters
        last_epoch = 0
        
        for e in self.policy_schedule:
            if iters <= e * self.num_iter_per_epoch:
                target_epoch = e
                break
            target_epoch = e
            last_epoch = e

        curr_schedule = self.policy_schedule[target_epoch]
        if curr_schedule['policy'] == 'warm':
            factor = (iters - last_epoch * self.num_iter_per_epoch) / ((target_epoch - last_epoch) * self.num_iter_per_epoch)
        elif curr_schedule['policy'] == 'cosine':
            factor = self.cosineLR(
                iters - last_epoch * self.num_iter_per_epoch, 
                curr_schedule['T_0'], 
                curr_schedule['T_MULT'], 
                self.num_iter_per_epoch
            ) * curr_schedule['decay'] * self.permanent_factor
        elif curr_schedule['policy'] == 'step':
            factor = curr_schedule['gamma'] ** curr_schedule['power']
            self.permanent_factor = factor
        elif curr_schedule['policy'] == 'none':
            factor = 1.0 * self.permanent_factor
        
        self._adjust_lr(factor)

    def _adjust_lr(self, factor):
        self.monitor_lrs = []
        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            param_group['lr'] = base_lr * factor
            self.monitor_lrs.append(base_lr * factor)

    def policy_parser(self, raw_policy):
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
                policy[target_epoch+duration] = {'policy':'step', 'gamma': float(parsed[2]), 'power': float(parsed[3])}
            elif parsed[0] == 'none':
                policy[target_epoch+duration] = {'policy':'none'}
            else:
                raise TypeError
            target_epoch += duration
        return policy

    @staticmethod
    def calc_cos_epochs(base, period, cycles):
        n_epochs = 0
        for i in range(cycles):
            n_epochs += period * (base ** i)
        return n_epochs
    
    @staticmethod
    def cosineLR(batch_idx, T_0, T_mult, num_iter_per_epoch):      
               
        restart_period = T_0 * num_iter_per_epoch

        while batch_idx/restart_period > 1.:
            batch_idx = batch_idx - restart_period
            restart_period = restart_period * T_mult

        radians = math.pi*(batch_idx/restart_period)

        return 0.5*(1.0 + math.cos(radians))



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