from src.trainer import *
from src.solver.solver import Solver
import logging
logger = logging.getLogger("logger")

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
        for sub_model in self.graph.sub_models:
            self.solvers[sub_model] = Solver(
                cfg, 
                [self.graph.sub_models[sub_model].named_parameters()], 
                lr=cfg.REID.CENTER_LOSS_LR, 
                lr_policy="none",
                opt_name="SGDW"
            )
        self.solvers['main'] = Solver(
            cfg, [self.graph.model.named_parameters()])
        self.graph.to_gpu()
        self.activate()

class HarmAttnReIDTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(HarmAttnReIDTrainer, self).__init__(cfg)
        self.solvers['main'] = Solver(
            cfg, [self.graph.model.named_parameters()])
        self.graph.to_gpu()
        self.activate()

class IAPReIDTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(IAPReIDTrainer, self).__init__(cfg)
        self.solvers['main'] = Solver(
            cfg, [self.graph.model.named_parameters()])
        self.graph.to_gpu()
        if cfg.SOLVER.MODEL_FREEZE_PEROID > 0:
            for n, p in self.graph.model.named_parameters():
                if 'iap' not in n:
                    p.requires_grad_(False)
                if p.requires_grad:
                    logger.info(f"{n} is trainable")
        self.activate()
