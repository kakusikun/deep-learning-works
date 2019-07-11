import os
from tqdm import tqdm
import torch
import glog

class ImageNetEngine():
    def __init__(self, cfg, criteria, opt, tdata, vdata, show, manager):
        self.cfg = cfg
        self.cores = manager.models
        self.criteria = criteria
        self.opt = opt
        self.tdata = tdata
        self.vdata = vdata
        self.show = show
        self.manager = manager

        self.iter = 0
        self.epoch = 0
        self.max_epoch = cfg.SOLVER.MAX_EPOCHS
        self.use_gpu = False   
        self.loss = 0.0
        self.train_accu = 0.0
        self.best_accu = 0.0
        self.accu = 0.0
        self.terminate = False

    def _start(self):
        if self.opt.findLR:
            glog.info("LR range test start")
        else:
            glog.info("Training start")
        self.iter = self.cfg.SOLVER.START_EPOCH * len(self.tdata)
        self.epoch = self.cfg.SOLVER.START_EPOCH
        self._check_gpu()        

    def _train_epoch_start(self): 
        self.epoch += 1
        glog.info("Epoch {} start".format(self.epoch))

        for core in self.cores.keys():
            self.cores[core].train()  

    def _eval_epoch_start(self): 
        for core in self.cores.keys():
            self.cores[core].eval()         

    def _train_iter_start(self):
        self.iter += 1
        self.opt._iter_start(self.iter, self.epoch)

    def _eval_iter_start(self):
        raise NotImplementedError
            
    def _train_iter_end(self):           
        self.show.add_scalar('train/loss', self.loss, self.iter)
        self.show.add_scalar('train/lr', self.opt.lr * self.opt.annealing_mult, self.iter)
        self.show.add_scalar('train/accuracy', self.train_accu, self.iter)      

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
        self.show.add_scalar('val/loss', self.loss, self.epoch)
        self.show.add_scalar('val/accuracy', self.best_accu, self.epoch)

    def _train_once(self):
        for batch in tqdm(self.tdata, desc="Epoch[{}/{}]".format(self.epoch, self.max_epoch)):

            if self.terminate:
                break

            self._train_iter_start()

            images, labels = batch
            if self.use_gpu:
                images, labels = images.cuda(), labels.cuda()
            
            outputs = self.cores['main'](images)
            for core in self.cores.keys():
                if core != 'main':
                    outputs = self.cores[core](outputs)
            
            loss = self.criteria(outputs, labels)

            if torch.isnan(loss):
                self.terminate = True                

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            _, prediction = outputs.max(1)
            count = outputs.size(0)
            correct = prediction.eq(labels).sum().item()  

            self.loss = loss.item()
            self.train_accu = correct * 100.0 / count 

            self._train_iter_end()

    def Train(self):            
        self._start()

        for i in range(self.max_epoch):
            if self.terminate:
                glog.info("Engine stops")
                break
            self._train_epoch_start()
            self._train_once()

            if not self.opt.findLR:
                self._evaluate()

    def Inference(self):
        glog.info("Epoch {} evaluation start".format(self.epoch))
        count = 0
        correct = 0
        
        with torch.no_grad():
            self._eval_epoch_start()
            for batch in tqdm(self.vdata, desc="Validation"): 
                images, labels = batch

                if self.use_gpu:
                    images, labels = images.cuda(), labels.cuda()
                
                outputs = self.cores['main'](images)

                for core in self.cores.keys():
                    if core != 'main':
                        outputs = self.cores[core](outputs)

                _, prediction = outputs.max(1)

                count += outputs.size(0)

                correct += prediction.eq(labels).sum().item() 
          
        self.accu = correct * 100.0 / count

        glog.info("Evaluation ends, accuracy {:.4f}".format(self.accu))

    def _evaluate(self):
        glog.info("Epoch {} evaluation start".format(self.epoch))
        count = 0
        correct = 0
        
        with torch.no_grad():
            self._eval_epoch_start()
            for batch in tqdm(self.vdata, desc="Validation"): 
                images, labels = batch

                if self.use_gpu:
                    images, labels = images.cuda(), labels.cuda()
                
                outputs = self.cores['main'](images)

                for core in self.cores.keys():
                    if core != 'main':
                        outputs = self.cores[core](outputs)

                loss = self.criteria(outputs, labels)

                _, prediction = outputs.max(1)

                count += outputs.size(0)

                correct += prediction.eq(labels).sum().item() 

                self.loss = loss.item()
          
        self.accu = correct * 100.0 / count

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

