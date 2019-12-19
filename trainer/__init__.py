from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from database.loader_factory import get_loader
from engine.engine_factory import get_engine
from manager.manager_factory import get_manager

from solver.optimizer import Solver
from visualizer.visualizer import Visualizer

import torch.nn as nn

class BaseTrainer():
    def __init__(self, cfg):
        self.loader = get_loader(cfg.DB.LOADER)(cfg)
        self.manager = get_manager(cfg.MANAGER)(cfg)
        self.manager.use_multigpu()

        self.solvers = {}  
        self.solvers['model'] = Solver(cfg, self.manager.model.named_parameters())
        self.visualizer = Visualizer(cfg)

    def activate(self, cfg):
        self.engine = get_engine(cfg.ENGINE)(cfg, self.solvers, self.loader, self.visualizer, self.manager)  
    
    def train(self):
        self.engine.Train()

    def test(self):
        self.engine.Evaluate()


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

