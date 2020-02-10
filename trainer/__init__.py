from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from database.loader_factory import get_loader
from engine.engine_factory import get_engine
from manager.manager_factory import get_manager

from solver.optimizer import Solver
from visualizer.visualizer import Visualizer
import logging
logger = logging.getLogger("logger")

import torch.nn as nn

class BaseTrainer():
    def __init__(self, cfg):
        # TODO: check config after building the loader
        self.cfg = cfg
        if self.cfg.SOLVER.LR_POLICY == 'cosine':
            adjusted_epochs, num_restart = self.calc_restart_maxepochs(self.cfg.SOLVER.WARMRESTART_MULTIPLIER, self.cfg.SOLVER.WARMRESTART_PERIOD, self.cfg.SOLVER.MAX_EPOCHS)
            logger.info(f"Max epochs is adjusted : {self.cfg.SOLVER.MAX_EPOCHS} => {adjusted_epochs} with {num_restart} restarts")
            self.cfg.SOLVER.MAX_EPOCHS = adjusted_epochs
        self.loader = get_loader(cfg.DB.LOADER)(cfg)
        self.manager = get_manager(cfg.MANAGER)(cfg)
        self.manager.use_multigpu()
        self.acc = 0.0

        self.solvers = {}  
        self.solvers['model'] = Solver(cfg, self.manager.model.named_parameters())

        self.visualizer = None
        if cfg.IO:
            self.visualizer = Visualizer(cfg)

    def activate(self, cfg):
        self.resume()
        self.engine = get_engine(cfg.ENGINE)(cfg, self.solvers, self.loader, self.visualizer, self.manager)
    
    def train(self):
        self.engine.Train()
        self.acc = self.engine.best_accu

    def test(self):
        self.engine.Evaluate()
        self.acc = self.engine.accu

    def resume(self):
        if self.cfg.RESUME:
            logger.info("Resuming from {}".format(self.cfg.RESUME))
            self.manager.load(self.cfg.RESUME)
            for solver in self.solvers:
                self.solvers[solver].load(self.cfg.RESUME, solver)            
        else:
            logger.info("Training model from scratch")

    @staticmethod
    def calc_restart_maxepochs(base, period, epochs):
        adjusted_epoch = 0
        i = 0
        while True:    
            gap = period * base ** i
            if adjusted_epoch + gap > epochs:
                break
            adjusted_epoch += gap
            i += 1
        return adjusted_epoch, i-1



class CenterKPTrainer(BaseTrainer):
    def __init__(self, cfg):
        assert cfg.INPUT.TEST_BS == 1
        super(CenterKPTrainer, self).__init__(cfg)
        self.activate(cfg)

class CenterTrainer(BaseTrainer):
    def __init__(self, cfg):
        assert cfg.INPUT.TEST_BS == 1
        super(CenterTrainer, self).__init__(cfg)        
        self.activate(cfg)

class ImagenetTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(ImagenetTrainer, self).__init__(cfg)
        self.activate(cfg)

class ReIDTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(ReIDTrainer, self).__init__(cfg)
        for submodel in self.manager.submodels:
            self.solvers[submodel] = Solver(cfg, self.manager.submodels[submodel].named_parameters(), _lr=cfg.SOLVER.CENTER_LOSS_LR, _name="SGD", _lr_policy="none")
        self.activate(cfg)

