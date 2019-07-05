import os
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
from tools.eval_reid_metrics import evaluate
from model.utility import GradNorm
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
        # self.weight_handler = GradNorm(cfg, self.cores['main'].l_features.conv.weight, device=cfg.MODEL.NUM_GPUS) 
        # self.weights = 0.0

    def _start(self):
        if self.opt.findLR:
            glog.info("LR range test start")
        else:
            glog.info("Training start")
        self.iter = self.cfg.OPTIMIZER.START_EPOCH * len(self.tdata)
        self.epoch = self.cfg.OPTIMIZER.START_EPOCH
        self._check_gpu()      

        # self.weight_handler.weight_initialize(self.cores, self.tdata, self.use_gpu) 

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
        # if self.epoch == self.opt.cum_epoch:
        #     self.weight_handler.need_initial = True
        self.opt._iter_start(self.iter, self.epoch)

    def _eval_iter_start(self):
        raise NotImplementedError
            
    def _train_iter_end(self):          
        self.show.add_scalar('train/glob_loss', self.loss[0], self.iter)
        self.show.add_scalar('train/center_loss', self.loss[1], self.iter)
        self.show.add_scalar('train/gpush_loss', self.loss[2], self.iter)
        self.show.add_scalar('train/push_loss', self.loss[3], self.iter)
        # self.show.add_scalar('train/glob_weight', self.weights[0], self.iter)
        # self.show.add_scalar('train/center_weight', self.weights[1], self.iter)
        # self.show.add_scalar('train/gpush_weight', self.weights[2], self.iter)
        # self.show.add_scalar('train/push_weight', self.weights[3], self.iter)
        self.show.add_scalar('train/lr', self.opt.lr * self.opt.annealing_mult, self.iter)

    def _eval_iter_end(self):           
        raise NotImplementedError

    def _train_epoch_end(self):
        if self.epoch % self.cfg.OPTIMIZER.LOG_FREQ == 0:
            if isinstance(self.cores['local_loss'], torch.nn.DataParallel): 
                local_embeddings = self.cores['local_loss'].module.center.data
                glob_embeddings = self.cores['glob_loss'].module.weight.data        
            else:
                local_embeddings = self.cores['local_loss'].center.data
                glob_embeddings = self.cores['glob_loss'].weight.data        

            self.show.add_embedding(local_embeddings, global_step=self.epoch, tag="local_embedding")
            self.show.add_embedding(glob_embeddings, global_step=self.epoch, tag="global_embedding")

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

            local_loss = list(self.cores['local_loss'](local, labels))
            glob_loss = self.cores['glob_loss'](glob, labels)
            
            loss = torch.stack([glob_loss] + local_loss)
            # weights = self.weight_handler.weights.expand_as(loss.t()).t()            

            # loss = weights * loss

            lg, bs = loss.size()
            _, indice = loss[:3].sum(0).sort(descending=True)
            if self.use_gpu:
                mask = torch.zeros(loss.size(1)).cuda()
            else:
                mask = torch.zeros(loss.size(1))

            effective_idx = mask.scatter(0, indice[:bs//2], 1).expand_as(loss)  
            loss = loss[effective_idx == 1].view(lg, bs//2).mean(1)

            self.opt.before_backward()
            # self.weights = self.weight_handler.weights.tolist()
            loss.sum().backward()          
            self.opt.after_backward()    

            # if self.weight_handler.need_initial:
            #     self.weight_handler.weight_initialize(self.cores, self.tdata, self.use_gpu) 
            # self.weight_handler.loss_weight_backward(loss)


            self.loss = loss.tolist()

            self._train_iter_end()

    def Train(self):
        self._start()
        for i in range(self.max_epoch):
            self._train_epoch_start()
            self._train_once()
            self._train_epoch_end()
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
        

    def _check_gpu(self):

        if self.cfg.MODEL.NUM_GPUS > 0 and torch.cuda.is_available():
            self.use_gpu = True
            glog.info("{} GPUs available".format(torch.cuda.device_count()))
        
            if self.cfg.MODEL.NUM_GPUS > 1 and torch.cuda.device_count() > 1:
                for core in self.cores.keys():
                    self.cores[core] = torch.nn.DataParallel(self.cores[core]).cuda()
            else:
                for core in self.cores.keys():
                    self.cores[core] = self.cores[core].cuda()

