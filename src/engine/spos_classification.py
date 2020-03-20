from src.engine import *
from tqdm import tqdm

import time
from copy import deepcopy
import multiprocessing
from multiprocessing import Value
from ctypes import c_bool

from tools.spos_utils import recalc_bn, Evolution

class SPOSClassificationEngine(BaseEngine):
    def __init__(self, cfg, graph, loader, solvers, visualizer):
        super(SPOSClassificationEngine, self).__init__(cfg, graph, loader, solvers, visualizer)
        self.manager = multiprocessing.Manager()
        self.cand_pool = self.manager.list()
        self.lock = self.manager.Lock()
        self.finished = Value(c_bool, False)
        self.evolution = Evolution(cfg, graph, logger=logger)
        
    def _train_once(self):
        accus = []   
        for batch in tqdm(self.tdata, desc=f"TRAIN[{self.epoch}/{self.cfg.SOLVER.MAX_EPOCHS}]"):
            self._train_iter_start()
            for key in batch:
                batch[key] = batch[key].cuda()

            cand = None
            while cand is None:
                if len(self.cand_pool) > 0:
                    with self.lock:
                        cand = self.cand_pool.pop()
                else:
                    time.sleep(1)

            channel_choices = cand['channel_choices']
            block_choices = cand['block_choices']     
            self.visualizer.add_histogram('train/evolution/block_choices', self._choice2hist(block_choices), self.iter, np.arange(len(block_choices)))
            self.visualizer.add_histogram('train/evolution/channel_choices', self._choice2hist(channel_choices), self.iter, np.arange(len(channel_choices)))
            self.visualizer.add_scalar('train/evolution/flops', cand['flops'], self.iter)              
            self.visualizer.add_scalar('train/evolution/params', cand['param'], self.iter)                      
            outputs = self.graph.model(batch['inp'], block_choices, channel_choices)
            self.loss, self.losses = self.graph.loss_head(outputs, batch)
            accus.append((outputs.max(1)[1] == batch['target']).float().mean())        
            self._train_iter_end()    
        self.train_accu = self.tensor_to_scalar(torch.stack(accus).mean())
        self.finished.value = True

    def Train(self):
        self._start()
        while self.epoch < self.max_epoch:
            self._train_epoch_start()
            self.finished.value = False
            if self.epoch - self.cfg.SPOS.EPOCH_TO_SEARCH == 1:
                self._copy_weight()
            pool_process = multiprocessing.Process(target=self.evolution.maintain,
                args=[self.epoch - self.cfg.SPOS.EPOCH_TO_SEARCH, self.cand_pool, self.lock, self.finished, logger])

            pool_process.start()                                        
            self._train_once()
            pool_process.join()
            self._train_epoch_end()
            self.graph.save(self.graph.save_path, self.graph.model, self.graph.sub_models, self.solvers, self.epoch, 0.0)                    
            if self.epoch % self.cfg.EVALUATE_FREQ == 0:
                self._evaluate()
            if self.cfg.SOLVER.LR_POLICY == 'plateau' and self.cfg.SOLVER.MIN_LR >= self.solvers['model'].monitor_lr:
                logger.info(f"LR {self.solvers['model'].monitor_lr} is less than {self.cfg.SOLVER.MIN_LR}")
                break
        logger.info(f"Best accuracy {self.best_accu:.2f}")
            
    def _evaluate(self, eval=False):
        logger.info("Epoch {} evaluation start".format(self.epoch))
        title = "EVALUATE" if eval else f"TEST[{self.epoch}/{self.cfg.SOLVER.MAX_EPOCHS}]"
        accus = []        
        block_choices = self.graph.random_block_choices(self.epoch - self.cfg.SPOS.EPOCH_TO_SEARCH)
        channel_choices = self.graph.random_channel_choices(self.epoch - self.cfg.SPOS.EPOCH_TO_SEARCH)
        raw_model_state = deepcopy(self.graph.model.state_dict())
        recalc_bn(self.graph, block_choices, channel_choices, self.tdata, True)

        with torch.no_grad():
            self._eval_epoch_start()
            for batch in tqdm(self.vdata, desc=title): 
                for key in batch:
                    batch[key] = batch[key].cuda()
                outputs = self.graph.model(batch['inp'], block_choices, channel_choices)
                accus.append((outputs.max(1)[1] == batch['target']).float().mean())
          
        self.accu = self.tensor_to_scalar(torch.stack(accus).mean())   
        if not eval:
            self._eval_epoch_end()       
            self.graph.model.load_state_dict(raw_model_state) 

    def Evaluate(self):
        self._evaluate(eval=True)
        logger.info(self.accu)
        
    def _copy_weight(self):
        logger.info("Copy Weights from Nas Blocks")
        if isinstance(self.graph.model, nn.DataParallel):
            for m in self.graph.model.module.modules():
                if hasattr(m, 'copy_weight'):
                    # before_p = deepcopy(next(iter(m.parameters())))
                    m.copy_weight()
                    # after_p = next(iter(m.parameters()))
                    # assert (after_p == before_p).sum() == 0
        else:
            for m in self.graph.model.modules():
                if hasattr(m, 'copy_weight'):
                    # before_p = deepcopy(next(iter(m.parameters())))
                    m.copy_weight()
                    # after_p = next(iter(m.parameters()))
                    # assert (after_p == before_p).sum() == 0

    def _choice2hist(self, choice):
        hist = []
        for i, c in enumerate(choice, 1):
            for _ in range(c):
                hist.append(i)
        return np.array(hist)
