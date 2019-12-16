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
        for batch in tqdm(self.tdata, desc="Epoch[{}/{}]".format(self.epoch, self.max_epoch)):
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

            self.train_accu = (outputs.max(1)[1] == batch['target']).float().mean() 
           

    def _evaluate(self, eval=False):
        logger.info("Epoch {} evaluation start".format(self.epoch))
        accus = []        
        with torch.no_grad():
            self._eval_epoch_start()
            for batch in tqdm(self.vdata, desc="Validation"): 
                for key in batch:
                    batch[key] = batch[key].cuda()
                images = batch['inp']      
                
                outputs = self.core(images)

                accus.append((outputs.max(1)[1] == batch['target']).float().mean())
          
        self.accu = torch.stack(accus).mean()

        if not eval:
            self._eval_epoch_end()        

    def Evaluate(self):
        self._evaluate(eval=True)
        logger.info(self.accu)
        
