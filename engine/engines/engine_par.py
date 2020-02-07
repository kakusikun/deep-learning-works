import os
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from engine.base_engine import BaseEngine, data_prefetcher
from tools.eval_par_metrics import eval_par_accuracy
import numpy as np
import logging
logger = logging.getLogger("logger")
# recover = T.Compose([T.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225], std = [1/0.229,1/0.224,1/0.225])])

class PAREngine(BaseEngine):
    def __init__(self, cfg, opts, loader, show, manager):
        super(PAREngine, self).__init__(cfg, opts, loader, show, manager)

    def _train_iter_start(self):
        self.iter += 1
        for opt in self.opts:
            opt.lr_adjust(self.total_loss, self.iter)
            opt.zero_grad()

    def _train_iter_end(self):  
        for opt in self.opts:
            opt.step()
 
        self.visualizer.add_scalar('train/total_loss', self.total_loss, self.iter)              
        for i in range(len(self.each_loss)):
            self.visualizer.add_scalar('train/loss/{}'.format(self.manager.loss_name[i]), self.each_loss[i], self.iter)
        self.visualizer.add_scalar('train/accuracy', self.train_accu, self.iter)   
        for i in range(len(self.opts)):
            self.visualizer.add_scalar('train/opt/{}/lr'.format(i), self.opts[i].monitor_lr, self.iter)

    def _train_once(self):
        prefetcher = data_prefetcher(self.tdata)
        for _ in tqdm(range(len(self.tdata)), desc="Epoch[{}/{}]".format(self.epoch, self.max_epoch)):
            self._train_iter_start()

            batch = prefetcher.next()
            if batch is None:
                break
            images, target = batch  
        
            output = self.core(images) 
            self.total_loss, self.each_loss, accu = self.manager.loss_func(output, target)
            self.total_loss.backward()

            self._train_iter_end()

            self.total_loss = self.tensor_to_scalar(self.total_loss)
            self.each_loss = self.tensor_to_scalar(self.each_loss)

            self.train_accu = accu         

    def _evaluate(self, eval=False):
        logger.info("Epoch {} evaluation start".format(self.epoch))
        self._eval_epoch_start()
        outputs = []
        targets = []
        test_loss = []
        with torch.no_grad():
            for batch in tqdm(self.vdata, desc="Validation"):
                
                images, target = batch
                if self.use_gpu: images, target = images.cuda(), target.cuda()

                output = self.core(images)
                loss, _, _ = self.manager.loss_func(output, target)

                test_loss.append(loss.cpu())
                outputs.append(output.cpu())
                targets.append(target)
                
        self.test_loss = torch.Tensor(test_loss).mean()
        pt = torch.cat(outputs, 0)
        gt = torch.cat(targets, 0)
        
        if self.use_gpu:
            precs, recalls = eval_par_accuracy(pt.cpu().numpy(), gt.cpu().numpy())
        else:
            precs, recalls = eval_par_accuracy(pt.numpy(), gt.numpy())

        t_precs = precs.mean(axis=1)
        t_recalls = recalls.mean(axis=1)
        logger.info("Computing Precision and Recall")
        logger.info("Results ----------")
        for thresh in [25, 50, 75]:
            logger.info("Threshold: {:5}".format(thresh*0.01))
            logger.info("{:10}  |  Precision: {:.2f}  |  Recall: {:.2f}".format("Total", t_precs[thresh], t_recalls[thresh]))
            for i, attr in enumerate(self.manager.category_names):
                if (i+1) not in self.cfg.PAR.IGNORE_CAT:
                    logger.info("{:10}  |  Precision: {:.2f}  |  Recall: {:.2f}".format(attr, precs[thresh][i], recalls[thresh][i]))
            logger.info("##################")

        logger.info("------------------")

        if not eval:
            self.accu = t_precs[50]
            self._eval_epoch_end() 
        else:
            np.save("{}/par_prec.npy".format(self.cfg.OUTPUT_DIR), precs)
            np.save("{}/par_recall.npy".format(self.cfg.OUTPUT_DIR), recalls)

    def Evaluate(self):
        self._evaluate(eval=True)
