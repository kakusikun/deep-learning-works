import os
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from engine.engine import Engine
from tools.eval_reid_metrics import evaluate
import numpy as np
import logging
logger = logging.getLogger("logger")
# recover = T.Compose([T.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225], std = [1/0.229,1/0.224,1/0.225])])

class PAREngine(Engine):
    def __init__(self, cfg, opts, tdata, qdata, show, manager):
        super(PAREngine, self).__init__(cfg, opts, tdata, None, qdata, None, show, manager)
            
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
            
            output = self.core(images) 
            self.total_loss, self.each_loss, accu = self.manager.loss_func(output, target)
            self.total_loss.backward()

            self._train_iter_end()

            self.total_loss = self.tensor_to_scalar(self.total_loss)
            self.each_loss = self.tensor_to_scalar(self.each_loss)

            self.train_accu = accu         

    def _evaluate(self):
        logger.info("Epoch {} evaluation start".format(self.epoch))
        self._eval_epoch_start()
        outputs = []
        targets = []
        with torch.no_grad():
            for batch in tqdm(self.qdata, desc="Validation"):
                
                images, target = batch
                if self.use_gpu: images = images.cuda()
                
                output = self.core(images)
                outputs.append(output.cpu())
                targets.append(target)
                
        pt = torch.cat(outputs, 0)
        gt = torch.cat(targets, 0)
        
        TPR, FPR, total_precision = eval_par_accuracy(pt.numpy(), gt.numpy())

        self.accu = total_precision[50]

        logger.info("Computing Prec and Recall")
        logger.info("Results ----------")
        logger.info("ROC curve")
        for thresh in [0, 25, 50, 75]:
            logger.info("Threshold: {:<3}  |  Precision: {:.2f}  |  TPR: {:.2f}  |  FPR: {:.2f}".format(thresh*0.01, total_precision[thresh], TPR[thresh], FPR[thresh]))
        logger.info("------------------")

        

