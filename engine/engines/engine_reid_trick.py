import os
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from engine.engine import Engine, data_prefetcher
from tools.eval_reid_metrics import evaluate, eval_recall
import numpy as np
import logging
logger = logging.getLogger("logger")
# recover = T.Compose([T.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225], std = [1/0.229,1/0.224,1/0.225])])

class ReIDEngine(Engine):
    def __init__(self, cfg, opts, tdata, qdata, gdata, show, manager):
        super(ReIDEngine, self).__init__(cfg, opts, tdata, None, qdata, gdata, show, manager)

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
        prefetcher = data_prefetcher(self.tdata)
        for _ in tqdm(range(len(self.tdata)), desc="Epoch[{}/{}]".format(self.epoch, self.max_epoch)):
            self._train_iter_start()

            batch = prefetcher.next()
            if batch is None:
                break
            images, target, _ = batch            


            local, glob = self.core(images) 
            self.total_loss, self.each_loss = self.manager.loss_func(local, glob, target)
            self.total_loss.backward()

            for _loss in self.manager.loss_has_param:
                for param in _loss.parameters():
                    param.grad.data *= (1. / self.cfg.SOLVER.CENTER_LOSS_WEIGHT)

            self._train_iter_end()

            self.total_loss = self.tensor_to_scalar(self.total_loss)
            self.each_loss = self.tensor_to_scalar(self.each_loss)

            self.train_accu = (glob.max(1)[1] == target).float().mean()          

    def _evaluate(self, eval=False):
        logger.info("Epoch {} evaluation start".format(self.epoch))
        self._eval_epoch_start()
        with torch.no_grad():
            qf, q_pids, q_camids = [], [], []
            for batch in tqdm(self.qdata, desc="Validation"):
                
                imgs, pids, camids = batch
                if self.use_gpu: imgs = imgs.cuda()
                
                features = self.core(imgs)

                features = F.normalize(features)
                
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

                features = F.normalize(features)
                
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

        logger.info("Computing Recall")
        rs, confs, gts, fg = eval_recall(distmat, q_pids, g_pids, q_camids, g_camids)

        logger.info("Results ------------: {:>4} / {:>4} / {:>4}".format("Q0.5", "Q0.75", "Q0.95"))
        logger.info("Number of candidates: {:.2f} / {} / {}".format(np.quantile(rs, q = 0.5), np.quantile(rs, q = 0.75), np.quantile(rs, q = 0.95)))
        logger.info("          Confidence: {:.2f} / {:.2f} / {:.2f}".format(np.quantile(confs, q = 0.5), np.quantile(confs, q = 0.75), np.quantile(confs, q = 0.95)))
        logger.info("    Number of target: {:.2f} / {} / {}".format(np.quantile(gts, q = 0.5), np.quantile(gts, q = 0.75), np.quantile(gts, q = 0.95)))  
        logger.info("------------------")
        
        if eval:
            np.save(self.cfg.OUTPUT_DIR+"/rs.npy", rs)
            np.save(self.cfg.OUTPUT_DIR+"/confs.npy", confs)
            np.save(self.cfg.OUTPUT_DIR+"/gts.npy", gts)
            np.save(self.cfg.OUTPUT_DIR+"/filtered_gallery.npy", fg)

        if not eval:
            self.accu = cmc[0]
            self._eval_epoch_end()

        del qf, gf, distmat
        
    def Evaluate(self):
        self._evaluate(eval=True)
