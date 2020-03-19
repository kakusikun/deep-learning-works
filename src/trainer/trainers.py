from src.trainer import *
from src.solver.solver import Solver

class CenternetODTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(CenternetODTrainer, self).__init__(cfg)
        self.solvers['main'] = Solver(
            cfg, [self.graph.model.named_parameters()])
        self.graph.to_gpu()
        self.activate()        

class ImagenetTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(ImagenetTrainer, self).__init__(cfg)
        self.solvers['main'] = Solver(cfg, [self.graph.model.named_parameters()])
        self.graph.to_gpu()
        self.activate()

class SPOSClassificationTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(SPOSClassificationTrainer, self).__init__(cfg)
        self.solvers['main'] = Solver(
            cfg, [self.graph.model.named_parameters()])
        self.graph.to_gpu()
        self.activate()
        
class TrickReIDTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(TrickReIDTrainer, self).__init__(cfg)
        self.solvers['main'] = Solver(
            cfg, [self.graph.model.named_parameters()])
        for sub_model in self.graph.sub_models:
            self.solvers[sub_model] = Solver(
                cfg, 
                [self.graph.sub_models[sub_model].named_parameters()], 
                lr=cfg.REID.CENTER_LOSS_LR, 
                lr_policy="none",
                opt_name="SGDW"
            )
        self.graph.to_gpu()
        self.activate()
