import os
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from tools.eval_reid_metrics import evaluate
from model.utility import TripletLoss, FocalWeight, CrossEntropyLossLSR
import numpy as np
import glog

recover = T.Compose([T.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225], std = [1/0.229,1/0.224,1/0.225])])

class ReIDEngine():
    def __init__(self, cfg, criteria, opt, tdata, qdata, gdata, show, manager):
        self.cfg = cfg
        self.cores = manager.models
        self.local_criteria = TripletLoss()
        self.glob_criteria = CrossEntropyLossLSR(cfg.MODEL.NUM_CLASSES)
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
        self.q_p_n = None
        #  self.loss_weight = 0.0

    def _start(self):
        if self.opt.findLR:
            glog.info("LR range test start")
        else:
            glog.info("Training start")
        self.iter = self.cfg.OPTIMIZER.START_EPOCH * len(self.tdata)
        self.epoch = self.cfg.OPTIMIZER.START_EPOCH

        self._check_gpu()      
        #  self.weight_handler = FocalWeight(self.cfg.OPTIMIZER.NUM_LOSSES, use_gpu=self.use_gpu) 

    def _eval_epoch_start(self): 
        for core in self.cores.keys():
            self.cores[core].eval()  

    def _train_epoch_start(self): 
        self.epoch += 1
        glog.info("Epoch {} start".format(self.epoch))

        for core in self.cores.keys():
            self.cores[core].train() 

        #  if self.epoch == 1:
            #  self.weight_handler.weight_initialize(self.cores, self.tdata, self.local_criteria, self.glob_criteria)
  
    def _train_iter_start(self):
        self.iter += 1
        self.opt._iter_start(self.iter, self.epoch)

    def _eval_iter_start(self):
        raise NotImplementedError
            
    def _train_iter_end(self):                
        self.show.add_scalar('train/id_loss', self.loss[0], self.iter)
        self.show.add_scalar('train/triplet_loss', self.loss[1], self.iter)
        self.show.add_scalar('train/center_loss', self.loss[2], self.iter)
        self.show.add_image('train/q_p_n', self.q_p_n, self.iter)
        self.show.add_scalar('train/rank1', self.train_rank1, self.iter)      
        self.show.add_scalar('train/accuracy', self.train_accu, self.iter)      
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

            images = images[labels > 0,:]
            labels = labels[labels > 0]
            
            local, glob = self.cores['main'](images)

            triplet_loss, self.train_rank1, q_idx, hard_p_idx, hard_n_idx = self.local_criteria(local, labels)
            center_loss = self.cores['center_loss'](local, labels)
            glob_output = self.cores['id_feat'](glob, labels)
            id_loss = self.glob_criteria(glob_output, labels)

            self.train_accu = (glob_output.max(1)[1] == labels).float().mean()            

            #  loss = torch.stack([id_loss, triplet_loss])    

            #  focal_weight = self.weight_handler.get_loss_weight(loss)
            #  final_loss = focal_weight[0] * id_loss + focal_weight[1] * triplet_loss + 0.0005 * center_loss
            final_loss = id_loss + triplet_loss + 0.0005 * center_loss

            self.loss = [id_loss.item(), triplet_loss.item(), center_loss.item()]
            #  self.loss_weight = focal_weight.tolist()

            self.opt.before_backward()
            final_loss.backward()          
            self.opt.after_backward()                    

            query = recover(images[q_idx].squeeze())
            hard_p = recover(images[hard_p_idx].squeeze())
            hard_n = recover(images[hard_n_idx].squeeze())
            self.q_p_n = torchvision.utils.make_grid([query, hard_p, hard_n], nrow = 1)

            self._train_iter_end()

    def Train(self):
        self._start()
        for i in range(self.max_epoch):
            self._train_epoch_start()
            self._train_once()
            if not self.opt.findLR and self.epoch % self.cfg.OPTIMIZER.EVALUATE_FREQ == 0:
                self._evaluate()

    def Inference(self):
        self._check_gpu()      
        glog.info("Evaluation start")
        self._eval_epoch_start()
        with torch.no_grad():
            qf, q_pids, q_camids = [], [], []
            for batch in tqdm(self.qdata, desc="Validation"):
                
                imgs, pids, camids = batch
                if self.use_gpu:
                    imgs = imgs.cuda()
                
                _, features = self.cores['main'](imgs)

                features = F.normalize(features)
                
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

                features = F.normalize(features)
                
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

        del qf, gf, distmat

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

                features = F.normalize(features)
                
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

                features = F.normalize(features)
                
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

            self.local_criteria = self.local_criteria.cuda()

