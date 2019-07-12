import os
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
import numpy as np
import glog

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
        self.use_gpu = False   
        self.train = True  
        self.total_loss = 0.0
        self.each_loss = None
        self.train_accu = 0.0
        self.best_accu = 0.0
        self.accu = 0.0

    def _start(self):
        glog.info("Training start")
        self.iter = self.cfg.SOLVER.START_EPOCH * len(self.tdata)
        self.epoch = self.cfg.SOLVER.START_EPOCH

        self._check_gpu()      

    def _train_epoch_start(self): 
        self.epoch += 1
        glog.info("Epoch {} start".format(self.epoch))

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
        if self.cfg.EVALUATE == "":
            glog.info("Epoch {} evaluation ends, accuracy {:.4f}".format(self.epoch, self.accu))
            if self.accu > self.best_accu:
                glog.info("Save checkpoint, with {:.4f} improvement".format(self.accu - self.best_accu))
                self.manager.save_model(self.epoch, self.opts, self.accu)
                self.best_accu = self.accu
            self.show.add_scalar('val/accuracy', self.best_accu, self.epoch)
        else:
            glog.info("Evaluation ends, accuracy {:.4f}".format(self.accu))

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

    def _check_gpu(self):
        if self.cfg.MODEL.NUM_GPUS > 0 and torch.cuda.is_available():
            self.use_gpu = True
            glog.info("{} GPUs available".format(torch.cuda.device_count()))
        
            if self.cfg.MODEL.NUM_GPUS > 1 and torch.cuda.device_count() > 1:
                self.core = torch.nn.DataParallel(self.core).cuda()
            else:
                self.core = self.core.cuda()

    @staticmethod
    def tensor_to_scalar(tensor):
        if isinstance(tensor, list):
            scalar = []
            for _tensor in tensor:
                scalar.append(_tensor.item())
        else:
            scalar = tensor.item()
        return scalar