import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from engine.engine import Engine, data_prefetcher
import numpy as np
import logging
logger = logging.getLogger("logger")

class ImageNetEngine(Engine):
    def __init__(self, cfg, opts, tdata, vdata, show, manager):
        super(ImageNetEngine, self).__init__(cfg, opts, tdata, vdata, None, None, show, manager)

    def _train_iter_start(self):
        self.iter += 1
        for opt in self.opts:
            opt.lr_adjust(self.total_loss, self.iter)
            opt.zero_grad()
   
    def _train_iter_end(self):           
        for opt in self.opts:
            opt.step()

        self.show.add_scalar('train/total_loss', self.total_loss, self.iter)              
        for i in range(len(self.each_loss)):
            self.show.add_scalar('train/loss/{}'.format(self.manager.loss_name[i]), self.each_loss[i], self.iter)
        self.show.add_scalar('train/accuracy', self.train_accu, self.iter)   
        for i in range(len(self.opts)):
            self.show.add_scalar('train/opt/{}/lr'.format(i), self.opts[i].monitor_lr, self.iter) 

    def _train_once(self):
        for batch in tqdm(self.tdata, desc="Epoch[{}/{}]".format(self.epoch, self.max_epoch)):
            self._train_iter_start()

            images, target = batch
            if self.use_gpu: images, target = images.cuda(), target.cuda()
            
            outputs = self.core(images)

            self.total_loss, self.each_loss = self.manager.loss_func(outputs, target)
            self.total_loss.backward()

            self._train_iter_end()     

            self.total_loss = self.tensor_to_scalar(self.total_loss)
            self.each_loss = self.tensor_to_scalar(self.each_loss)

            self.train_accu = (outputs.max(1)[1] == target).float().mean() 
           

    def _evaluate(self):
        logger.info("Epoch {} evaluation start".format(self.epoch))
        accus = []        
        with torch.no_grad():
            self._eval_epoch_start()
            for batch in tqdm(self.vdata, desc="Validation"): 
                images, target = batch
                if self.use_gpu: images, target = images.cuda(), target.cuda()
                
                outputs = self.core(images)

                accus.append((outputs.max(1)[1] == target).float().mean())
          
        self.accu = torch.stack(accus).mean()

        self._eval_epoch_end()        

