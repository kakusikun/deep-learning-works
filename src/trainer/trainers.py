from src.trainer import *
from src.solver.solver import Solver

class CenternetODTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(CenternetODTrainer, self).__init__(cfg)   
        self.activate()        

class ImagenetTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(ImagenetTrainer, self).__init__(cfg)
        self.activate()

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


class ReIDTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(ReIDTrainer, self).__init__(cfg)
        for submodel in self.manager.submodels:
            self.solvers[submodel] = Solver(cfg, self.manager.submodels[submodel].named_parameters(), _lr=cfg.SOLVER.CENTER_LOSS_LR, _name="SGD", _lr_policy="none")
        self.activate(cfg)