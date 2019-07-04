import torch
from torch.optim.optimizer import Optimizer, required



class MySGD(Optimizer):  
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))    
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MySGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MySGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(lr, d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(lr, d_p)

                    if nesterov:
                        if 'nesterov_buffer' not in param_state:
                            nbuf = param_state['nesterov_buffer'] = torch.zeros_like(p.data)
                            nbuf.add_(d_p)
                            buf.add_(lr * momentum, d_p)
                        else:
                            nbuf = param_state['nesterov_buffer']
                            buf.add_(lr * momentum, d_p).add_(-lr * momentum, nbuf)
                            nbuf.mul_(0).add_(d_p)
                    d_p = buf
                else:
                    d_p.mul_(lr)

                if weight_decay != 0:
                    p.data.add_(-weight_decay, p.data).add_(-d_p)
                else:
                    p.data.add_(-d_p)                

        return loss



#https://arxiv.org/abs/1905.11286
#Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks
class NovoGrad(Optimizer):  
        def __init__(self, params, lr=1e-3, betas=(0.95, 0.98), eps=1e-8,
                 weight_decay=5*1e-4, amsgrad=False):
            if not 0.0 <= lr:
                raise ValueError("Invalid learning rate: {}".format(lr))
            if not 0.0 <= eps:
                raise ValueError("Invalid epsilon value: {}".format(eps))
            if not 0.0 <= betas[0] < 1.0:
                raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
            if not 0.0 <= betas[1] < 1.0:
                raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
            defaults = dict(lr=lr, betas=betas, eps=eps,
                            weight_decay=weight_decay, amsgrad=amsgrad)
            super(NovoGrad, self).__init__(params, defaults)

        def __setstate__(self, state):
            super(NovoGrad, self).__setstate__(state)
            for group in self.param_groups:
                group.setdefault('amsgrad', False)

        def step(self, closure=None):
            """Performs a single optimization step.

            Arguments:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
            """
            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    
                    grad_norm = grad.norm()

                    amsgrad = group['amsgrad']

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['vt'] = grad_norm ** 2

                        state['mt'] = grad.div(grad_norm).add(group['weight_decay'], p.data)
                        
                        if amsgrad:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_vt'] = torch.zeros_like(grad_norm)

                        vt, mt = state['vt'], state['mt']

                    else:
                        vt, mt = state['vt'], state['mt']

                        if amsgrad:
                            max_vt = state['max_vt']

                        beta1, beta2 = group['betas']

                        vt.mul_(beta2).addcmul_(1 - beta2, grad_norm, grad_norm)

                        if amsgrad:
                            torch.max(max_vt, vt, out=max_vt)
                            denom = max_vt.sqrt().add_(group['eps'])
                        else:
                            denom = vt.sqrt().add(group['eps'])

                        mt.mul_(beta1).add_(grad.div(denom)).add_(group['weight_decay'], p.data)
                                 
                    p.data.add_(-group['lr'], mt)

            return loss