import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from engine.base_engine import BaseEngine
import numpy as np
import logging
logger = logging.getLogger("logger")

class ImageNetEngine(BaseEngine):
    def __init__(self, cfg, solvers, loader, show, manager):
        super(ImageNetEngine, self).__init__(cfg, solvers, loader, show, manager)
        
    def _train_once(self):
        # for batch in tqdm(self.tdata, desc="Epoch[{}/{}]".format(self.epoch, self.max_epoch), position=0, leave=True):
        for i, batch in enumerate(self.tdata):
            self._train_iter_start()
            for key in batch:
                batch[key] = batch[key].cuda()
            images = batch['inp']            
            outputs = self.core(images)

            self.total_loss, self.each_loss = self.manager.loss_func(outputs, batch)
            self.total_loss.backward()

            self._train_iter_end()     

            self.total_loss = self.tensor_to_scalar(self.total_loss)
            self.each_loss = self.tensor_to_scalar(self.each_loss)

            self.train_accu = self.tensor_to_scalar((outputs.max(1)[1] == batch['target']).float().mean())
            if i % 10 == 0:
                logger.info(f"Epoch [{self.epoch:03}/{self.max_epoch:03}]   Step [{i:04}/{self.cfg.SOLVER.ITERATIONS_PER_EPOCH:04}]   loss {self.total_loss:3.3f}   accu {self.train_accu:3.3f}")
           

    def _evaluate(self, eval=False):
        logger.info("Epoch {} evaluation start".format(self.epoch))
        accus = []        
        with torch.no_grad():
            self._eval_epoch_start()
            # for batch in tqdm(self.vdata, desc="Validation", position=0, leave=True): 
            for batch in self.vdata: 
                for key in batch:
                    batch[key] = batch[key].cuda()
                images = batch['inp']      
                
                outputs = self.core(images)

                accus.append((outputs.max(1)[1] == batch['target']).float().mean())
          
        self.accu = self.tensor_to_scalar(torch.stack(accus).mean())    

        if not eval:
            self._eval_epoch_end()        

    def Evaluate(self):
        self._evaluate(eval=True)
        logger.info(self.accu)
        
