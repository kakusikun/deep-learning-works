import os
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from engine.base_engine import BaseEngine, data_prefetcher
from tools.eval_reid_metrics import evaluate
import numpy as np
import logging
logger = logging.getLogger("logger")
# recover = T.Compose([T.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225], std = [1/0.229,1/0.224,1/0.225])])

class MAREngine(BaseEngine):
    def __init__(self, cfg, opts, trtdata, tdata, qdata, gdata, show, manager):
        super(MAREngine, self).__init__(cfg, opts, tdata, None, qdata, gdata, show, manager)
        self.trtdata = trtdata
        self.trt_prefetcher = data_prefetcher(self.trtdata)

    def _train_iter_start(self):
        self.iter += 1
        for opt in self.opts:
            opt.lr_adjust(self.total_loss, self.iter)
            opt.zero_grad()

    def _train_iter_end(self):  
        for opt in self.opts:
            opt.step()

        total_loss = self.tensor_to_scalar(self.total_loss)
        each_loss = self.tensor_to_scalar(self.each_loss)
        self.show.add_scalar('train/total_loss', total_loss, self.iter)              
        for i in range(len(self.each_loss)):
            self.show.add_scalar('train/loss/{}'.format(self.manager.loss_name[i]), each_loss[i], self.iter)
        self.show.add_scalar('train/accuracy', self.train_accu, self.iter)   
        for i in range(len(self.opts)):
            self.show.add_scalar('train/opt/{}/lr'.format(i), self.opts[i].monitor_lr, self.iter)

    def _train_once(self):
        src_prefetcher = data_prefetcher(self.tdata)                
        for _ in tqdm(range(len(self.tdata)), desc="Epoch[{}/{}]".format(self.epoch, self.max_epoch)):
            self._train_iter_start()

            src_batch = src_prefetcher.next()
            if src_batch is None:
                break
            src_images, src_target, _ = src_batch            

            trt_batch = self.trt_prefetcher.next()
            if trt_batch is None:
                self.trt_prefetcher = data_prefetcher(self.trtdata)
                trt_batch = self.trt_prefetcher.next()                
            trt_images, _, trt_cams, trt_idx = trt_batch

            src_feat, src_sim = self.core(src_images) 
            trt_feat, trt_sim = self.core(trt_images)
            self.total_loss, self.each_loss = self.manager.loss_func(src_feat, trt_feat, src_sim, trt_sim, src_target, trt_cams, trt_idx, self.epoch)

            self.total_loss.backward()

            self._train_iter_end()

            self.train_accu = (src_sim.max(1)[1] == src_target).float().mean()          

    def _evaluate(self, eval=False):
        logger.info("Epoch {} evaluation start".format(self.epoch))
        self._eval_epoch_start()
        with torch.no_grad():
            qf, q_pids, q_camids = [], [], []
            for batch in tqdm(self.qdata, desc="Validation"):
                
                imgs, pids, camids = batch
                imgs = imgs.cuda()
                
                features = self.core(imgs)
                
                qf.append(features.cpu())
                q_pids.extend(pids)
                q_camids.extend(camids)

            qf = torch.cat(qf, 0)
            q_pids = np.asarray(q_pids)
            q_camids = np.asarray(q_camids)
            logger.info("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

            gf, g_pids, g_camids = [], [], []
            for batch in tqdm(self.gdata, desc="Validation"):
                
                imgs, pids, camids = batch
                if self.use_gpu: imgs = imgs.cuda()
                
                features = self.core(imgs)
                
                gf.append(features.cpu())
                g_pids.extend(pids)
                g_camids.extend(camids)

            gf = torch.cat(gf, 0)
            g_pids = np.asarray(g_pids)
            g_camids = np.asarray(g_camids)
            logger.info("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

        distmat =  1 - F.linear(qf, gf)
        distmat = distmat.numpy()

        logger.info("Computing CMC and mAP")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

        logger.info("Results ----------")
        logger.info("mAP: {:.1%}".format(mAP))
        logger.info("CMC curve")
        for r in [1, 5, 10, 20]:
            logger.info("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        logger.info("------------------")

        if not eval:
            self.accu = cmc[0]
            self._eval_epoch_end()

        del qf, gf, distmat
        
    def Evaluate(self):
        self._evaluate(eval=True)
