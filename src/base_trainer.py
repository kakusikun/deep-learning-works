import os
from copy import deepcopy
import numpy as np
import logging
logger = logging.getLogger("logger")

from src.factory.loader_factory import LoaderFactory
from src.factory.graph_factory import GraphFactory
from src.solver.solver import Solver
from src.factory.engine_factory import EngineFactory
from tools.tensorboard import Tensorboard
from tools.utils import print_config
try:
    from apex import amp
    APEX_IMPORTED = True
except:
    logger.info("Install nvidia apex first")
    APEX_IMPORTED = False

class BaseTrainer():
    def __init__(self, cfg):
        # TODO: check config after building the loader
        self.cfg = cfg
        self.loader = LoaderFactory.produce(cfg)
        # must be checked after the loader is built
        self.graph = GraphFactory.produce(cfg)                
        self.acc = 0.0
        self.solvers = {}          
        self.visualizer = None
        if cfg.IO:
            self.visualizer = Tensorboard(cfg)        
        
    def activate(self):
        self.resume()
        if self.cfg.APEX and APEX_IMPORTED:
            self.graph.model, self.solvers['main'].opt = amp.initialize(
                self.graph.model, 
                self.solvers['main'].opt,
                opt_level='O1',
                # keep_batchnorm_fp32=True
            )
                
        self.graph.to_gpus()
        self.engine = EngineFactory.produce(
            self.cfg, self.graph, self.loader, self.solvers, self.visualizer
        )
        if self.cfg.IO:
            print_config(self.cfg)        
    
    def train(self):
        self.engine.Train()
        self.acc = self.engine.best_accu

    def test(self):
        self.engine.Evaluate()
        self.acc = self.engine.accu

    def resume(self):
        if self.cfg.RESUME:
            logger.info("Resuming from {}".format(self.cfg.RESUME))
            self.graph.load(self.cfg.RESUME, solvers=self.solvers)
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
        if adjusted_epoch == 0:
            adjusted_epoch = 1
        if i == 0:
            i = 1
        return adjusted_epoch, i-1