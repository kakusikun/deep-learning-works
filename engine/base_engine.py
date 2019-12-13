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

class BaseEngine():
    def __init__(self, cfg, opts, loader, show, manager):
        self.cfg = cfg
        self.core = manager.model       
        self.loss_func = manager.loss_func
        self.opts = opts
        self.tdata = loader['train'] if 'train' in loader else None
        self.vdata = loader['val'] if 'val' in loader else None
        self.qdata = loader['val']['query'] if 'val' in loader and isinstance(loader['val'], dict) and 'query' in loader['val'] else None
        self.gdata = loader['val']['gallery'] if 'val' in loader and isinstance(loader['val'], dict) and 'gallery' in loader['val'] else None
        self.show = show
        self.manager = manager

        self.iter = 0
        self.epoch = 0
        self.max_epoch = cfg.SOLVER.MAX_EPOCHS
        self.use_gpu = manager.use_gpu 
        self.train = True  
        self.total_loss = 1e5
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
        if self.cfg.SOLVER.EVALUATE_FREQ > 0:
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
            if self.cfg.SOLVER.EVALUATE_FREQ > 0:
                if self.epoch % self.cfg.SOLVER.EVALUATE_FREQ == 0:
                    self._evaluate()
            else:
                self.manager.save_model(self.epoch, self.opts, 0.0)
            if self.cfg.SOLVER.LR_POLICY == 'plateau' and self.cfg.SOLVER.MIN_LR >= self.opts[-1].monitor_lr:
                logger.info("LR {} is less than the min LR {}".format(self.opts[0].monitor_lr, self.cfg.SOLVER.MIN_LR))
                break

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
            if tensor.is_cuda:
                scalar = tensor.cpu().detach().numpy().tolist()
            else:
                scalar = tensor.detach().numpy().tolist()
        else:
            scalar = tensor.item()
        return scalar

    

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        if isinstance(self.next_batch, list):
            self.next_batch_gpu = []
            for i in range(len(self.next_batch)):
                self.next_batch_gpu.append(torch.empty_like(self.next_batch[i], device='cuda'))
            # Need to make sure the memory allocated for next_* is not still in use by the main stream
            # at the time we start copying to next_*:
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                # more code for the alternative if record_stream() doesn't work:
                # copy_ will record the use of the pinned source tensor in this side stream.
                for i in range(len(self.next_batch)):
                    self.next_batch_gpu[i].copy_(self.next_batch[i], non_blocking=True)
                self.next_batch = self.next_batch_gpu

                # With Amp, it isn't necessary to manually convert data to half.
                # if args.fp16:
                #     self.next_input = self.next_input.half()
                # else:
        elif isinstance(self.next_batch, dict):
            self.next_batch_gpu = {}
            for key in self.next_batch.keys():
                self.next_batch_gpu[key] = torch.empty_like(self.next_batch[key], device='cuda')
            # Need to make sure the memory allocated for next_* is not still in use by the main stream
            # at the time we start copying to next_*:
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                # more code for the alternative if record_stream() doesn't work:
                # copy_ will record the use of the pinned source tensor in this side stream.
                for key in self.next_batch.keys():
                    self.next_batch_gpu[key].copy_(self.next_batch[key], non_blocking=True)
                self.next_batch = self.next_batch_gpu

                # With Amp, it isn't necessary to manually convert data to half.
                # if args.fp16:
                #     self.next_input = self.next_input.half()
                # else:
        else:
            raise TypeError
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if isinstance(batch, list):
            if batch is not None:
                for i in range(len(batch)):
                    batch[i].record_stream(torch.cuda.current_stream())
        elif isinstance(self.next_batch, dict):
            if batch is not None:
                for key in batch.keys():
                    batch[key].record_stream(torch.cuda.current_stream())
        else:
            raise TypeError

        self.preload()
        return batch
