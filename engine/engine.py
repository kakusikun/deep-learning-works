import os
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
import numpy as np
import logging
logger = logging.getLogger("logger")

class Engine():
    def __init__(self, cfg, opts, tdata, vdata, qdata, gdata, show, manager):
        self.cfg = cfg
        self.core = manager.model       
        self.loss_func = manager.loss_func
        self.opts = opts
        self.tdata = tdata
        self.vdata = vdata
        self.qdata = qdata
        self.gdata = gdata
        self.show = show
        self.manager = manager

        self.iter = 0
        self.epoch = 0
        self.max_epoch = cfg.SOLVER.MAX_EPOCHS
        self.use_gpu = next(self.core.parameters()).is_cuda 
        self.train = True  
        self.total_loss = 0.0
        self.each_loss = None
        self.train_accu = 0.0
        self.best_accu = 0.0
        self.min_loss = 1e5
        self.test_loss = 0.0
        self.accu = 0.0
        self.save_criterion = cfg.MODEL.SAVE_CRITERION

    def _start(self):
        logger.info("Training start")
        self.iter = self.cfg.SOLVER.START_EPOCH * len(self.tdata)
        self.epoch = self.cfg.SOLVER.START_EPOCH


    def _train_epoch_start(self): 
        self.epoch += 1
        logger.info("Epoch {} start".format(self.epoch))

        self.core.train() 
  
    def _train_iter_start(self):
        raise NotImplementedError

    def _train_iter_end(self):                
        raise NotImplementedError

    def _train_epoch_end(self):
        raise NotImplementedError

    def _eval_epoch_start(self): 
        self.core.eval() 

    def _eval_iter_start(self):
        raise NotImplementedError  

    def _eval_iter_end(self):           
        raise NotImplementedError

    def _eval_epoch_end(self):
        if self.save_criterion == 'loss':
            logger.info("Epoch {} evaluation ends, loss {:.4f}".format(self.epoch, self.test_loss))
            if self.min_loss > self.test_loss:
                logger.info("Save checkpoint, with {:.4f} improvement".format(self.min_loss - self.test_loss))
                self.manager.save_model(self.epoch, self.opts, self.test_loss)
                self.min_loss = self.test_loss
            self.show.add_scalar('val/loss', self.min_loss, self.epoch)
        else:
            logger.info("Epoch {} evaluation ends, accuracy {:.4f}".format(self.epoch, self.accu))
            if self.accu > self.best_accu:
                logger.info("Save checkpoint, with {:.4f} improvement".format(self.accu - self.best_accu))
                self.manager.save_model(self.epoch, self.opts, self.accu)
                self.best_accu = self.accu
            self.show.add_scalar('val/accuracy', self.best_accu, self.epoch)

    def _train_once(self):
        raise NotImplementedError

    def Train(self):
        self._start()
        for i in range(self.max_epoch):
            self._train_epoch_start()
            self._train_once()
            if self.epoch % self.cfg.SOLVER.EVALUATE_FREQ == 0:
                self._evaluate()

    def Inference(self):
        raise NotImplementedError

    def _evaluate(self):
        raise NotImplementedError        

    @staticmethod
    def tensor_to_scalar(tensor):
        if isinstance(tensor, list):
            scalar = []
            for _tensor in tensor:
                scalar.append(_tensor.item())
        elif isinstance(tensor, torch.Tensor) and tensor.dim() != 0:
            scalar = tensor.numpy().tolist()
        else:
            scalar = tensor.item()
        return scalar

    

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target