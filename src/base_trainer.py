import os
import numpy as np
import logging
logger = logging.getLogger("logger")

from src.factory.loader_factory import LoaderFactory
from src.factory.graph_factory import GraphFactory
from src.solver.solver import Solver
from src.factory.engine_factory import EngineFactory
from tools.tensorboard import Tensorboard

class BaseTrainer():
    def __init__(self, cfg):
        # TODO: check config after building the loader
        self.cfg = cfg
        self.loader = LoaderFactory.produce(cfg)
        # must be checked after the loader is built
        self._check_config()

        self.graph = GraphFactory.produce(cfg)
        self.graph.use_multigpu()
        self.acc = 0.0

        self.solvers = {}  
        self.solvers['main'] = Solver(cfg, self.graph.model.named_parameters())

        self.visualizer = None
        if cfg.IO:
            self.visualizer = Tensorboard(cfg)

        self.resume()
        
    def activate(self, cfg):
        self.engine = EngineFactory.produce(
            cfg, self.graph, self.loader, self.solvers, self.visualizer
        )
    
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
    
    def _check_config(self):
        if self.cfg.SOLVER.LR_POLICY == 'cosine':
            adjusted_epochs, num_restart = self.calc_restart_maxepochs(self.cfg.SOLVER.WARMRESTART_MULTIPLIER, self.cfg.SOLVER.WARMRESTART_PERIOD, self.cfg.SOLVER.MAX_EPOCHS)
            logger.info(f"Max epochs is adjusted : {self.cfg.SOLVER.MAX_EPOCHS} => {adjusted_epochs} with {num_restart} restarts")
            self.cfg.SOLVER.MAX_EPOCHS = adjusted_epochs
        self.cfg.SOLVER.ITERATIONS_PER_EPOCH = len(self.loader['train'])

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