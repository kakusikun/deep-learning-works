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
                        if i % 50 == 0:
                            logger.info('-' * 40)
                            logger.info(f"[Train] Block Choices: {cand['block_choices']}")
                            logger.info(f"[Train] Channel Choice: {cand['channel_choices']}")
                            logger.info(f"[Train] Flop: {cand['flops']:.2f}M, param: {cand['param']:.2f}M")
                else:
                    time.sleep(1)

            channel_choices = cand['channel_choices']
            block_choices = cand['block_choices']        
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
            self.finished.value = True
            pool_process = multiprocessing.Process(target=self.evolution.maintain,
                args=[self.epoch - self.cfg.SPOS.EPOCH_TO_SEARCH, self.cand_pool, self.lock, self.finished, logger])
            pool_process.start()                                        
            self._train_once()
            pool_process.join()
            self._train_epoch_end()
            
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
        block_choices = self.graph.random_block_choices()
        channel_choices = self.graph.random_channel_choices()
        raw_model_state = deepcopy(self.graph.model.state_dict())
        recalc_bn(self.graph, block_choices, channel_choices, self.tdata, True)

        with torch.no_grad():
            self._eval_epoch_start()
            for batch in tqdm(self.vdata, desc=title): 
                for key in batch:
                    batch[key] = batch[key].cuda()
                images = batch['inp']                      
                outputs = self.graph.model(images)
                accus.append((outputs.max(1)[1] == batch['target']).float().mean())
          
        self.accu = self.tensor_to_scalar(torch.stack(accus).mean())   
        if not eval:
            self._eval_epoch_end()       
            self.graph.model.load_state_dict(raw_model_state) 

    def Evaluate(self):
        self._evaluate(eval=True)
        logger.info(self.accu)
        
