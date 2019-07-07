import os
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
from tools.eval_reid_metrics import evaluate
import numpy as np
import glog

class ReIDEngine():
    def __init__(self, cfg, criteria, opt, tdata, qdata, gdata, show, manager):
        self.cfg = cfg
        self.cores = manager.models
        self.criteria = criteria
        self.opt = opt
        self.tdata = tdata
        self.qdata = qdata
        self.gdata = gdata
        self.show = show
        self.manager = manager

        self.iter = 0
        self.epoch = 0
        self.max_epoch = cfg.OPTIMIZER.MAX_EPOCHS
        self.use_gpu = False   
        self.train = True  
        self.loss = 0.0
        self.train_accu = 0.0
        self.best_accu = 0.0
        self.accu = 0.0
        self.phase = cfg.PHASE

    def _start(self):
        if self.opt.findLR:
            glog.info("LR range test start")
        else:
            glog.info("Training start")
        self.iter = self.cfg.OPTIMIZER.START_EPOCH * len(self.tdata)
        self.epoch = self.cfg.OPTIMIZER.START_EPOCH
        self._check_gpu()      


    def _eval_epoch_start(self): 
        for core in self.cores.keys():
            self.cores[core].eval()  

    def _train_epoch_start(self): 
        self.epoch += 1
        glog.info("Epoch {} start".format(self.epoch))

        for core in self.cores.keys():
            self.cores[core].train() 
  
    def _train_iter_start(self):
        self.iter += 1
        self.opt._iter_start(self.iter, self.epoch)

    def _eval_iter_start(self):
        raise NotImplementedError
            
    def _train_iter_end(self):                
        if self.phase == 2:
            self.show.add_scalar('train/glob_loss', self.loss[0], self.iter)
            self.show.add_scalar('train/push_loss', self.loss[1], self.iter)
            self.show.add_scalar('train/gpush_loss', self.loss[2], self.iter)
            # self.show.add_scalar('train/push_loss', self.loss[3], self.iter)
        else:
            self.show.add_scalar('train/glob_loss', self.loss, self.iter)
        self.show.add_scalar('train/lr', self.opt.lr * self.opt.annealing_mult, self.iter)

    def _eval_iter_end(self):           
        raise NotImplementedError

    def _train_epoch_end(self):
        raise NotImplementedError

    def _eval_epoch_end(self):
        glog.info("Epoch {} evaluation ends, accuracy {:.4f}".format(self.epoch, self.accu))
        if self.accu > self.best_accu:
            glog.info("Save checkpoint, with {:.4f} improvement".format(self.accu - self.best_accu))
            self.manager.save_model(self.epoch, self.opt, self.accu)
            self.best_accu = self.accu
        self.show.add_scalar('val/accuracy', self.best_accu, self.epoch)

    def _train_once(self):
        for batch in tqdm(self.tdata, desc="Epoch[{}/{}]".format(self.epoch, self.max_epoch)):

            self._train_iter_start()

            images, labels, _ = batch
            if self.use_gpu:
                images, labels = images.cuda(), labels.cuda()
            
            local, glob = self.cores['main'](images)

            glob_loss = self.cores['glob_loss'](glob, labels)

            if self.phase == 2:
                local_loss = list(self.cores['local_loss'](local, labels))
            
                loss = torch.stack([glob_loss.mean()] + local_loss)

                final_loss = loss.sum()

                self.loss = loss.tolist()

            else:
                final_loss = glob_loss.mean()       
                self.loss = final_loss.item()

            self.opt.before_backward()
            final_loss.backward()          
            self.opt.after_backward()                    

            self._train_iter_end()

    def Train(self):
        self._start()
        for i in range(self.max_epoch):
            self._train_epoch_start()
            self._train_once()
            if not self.opt.findLR and self.epoch % self.cfg.OPTIMIZER.EVALUATE_FREQ == 0:
                self._evaluate()

    def Inference(self):
        raise NotImplementedError

    def _evaluate(self):
        glog.info("Epoch {} evaluation start".format(self.epoch))
        self._eval_epoch_start()
        with torch.no_grad():
            qf, q_pids, q_camids = [], [], []
            for batch in tqdm(self.qdata, desc="Validation"):
                
                imgs, pids, camids = batch
                if self.use_gpu:
                    imgs = imgs.cuda()
                
                _, features = self.cores['main'](imgs)
                
                qf.append(features.cpu())
                q_pids.extend(pids)
                q_camids.extend(camids)

            qf = torch.cat(qf, 0)
            q_pids = np.asarray(q_pids)
            q_camids = np.asarray(q_camids)
            print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

            gf, g_pids, g_camids = [], [], []
            for batch in tqdm(self.gdata, desc="Validation"):
                
                imgs, pids, camids = batch
                if self.use_gpu:
                    imgs = imgs.cuda()
                
                _, features = self.cores['main'](imgs)
                
                gf.append(features.cpu())
                g_pids.extend(pids)
                g_camids.extend(camids)

            gf = torch.cat(gf, 0)
            g_pids = np.asarray(g_pids)
            g_camids = np.asarray(g_camids)
            print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

        distmat =  1 - F.linear(qf, gf)

        print("Computing CMC and mAP")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in [1, 5, 10, 20]:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")

        self.accu = cmc[0]

        self._eval_epoch_end()

        del qf, gf, distmat
        

    def _check_gpu(self):

        if self.cfg.MODEL.NUM_GPUS > 0 and torch.cuda.is_available():
            self.use_gpu = True
            glog.info("{} GPUs available".format(torch.cuda.device_count()))
        
            if self.cfg.MODEL.NUM_GPUS > 1 and torch.cuda.device_count() > 1:
                for core in self.cores.keys():
                    if core == 'main':
                        self.cores[core] = torch.nn.DataParallel(self.cores[core]).cuda()
                    else:
                        self.cores[core] = self.cores[core].cuda()
            else:
                for core in self.cores.keys():
                    self.cores[core] = self.cores[core].cuda()

