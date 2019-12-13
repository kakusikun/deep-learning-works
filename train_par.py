import argparse
import shutil
import sys

from config.config_factory import _C as cfg
from config.config_factory import build_output
from data.build_loader import build_par_loader
from engine.engines.engine_par import PAREngine
from solver.optimizer import Solver
from visualizer.visualizer import Visualizer
from manager.base_managers.manager_par import PARManager, SinglePARManager
from tools.logger import setup_logger
import torch.nn as nn

def train(cfg): 

    train_loader, val_loader = build_par_loader(cfg)

    manager = PARManager(cfg) if cfg.PAR.SELECT_CAT == -1 else SinglePARManager(cfg)

    manager.use_multigpu()

    cfg.SOLVER.ITERATIONS_PER_EPOCH = len(train_loader)

    opts = []    
    opts.append(Solver(cfg, manager.model.named_parameters()))

    visualizer = Visualizer(cfg)
    
    engine = PAREngine(cfg, opts, train_loader, val_loader, visualizer, manager)  
    if cfg.EVALUATE: 
        engine.Evaluate()
        sys.exit(1)
    engine.Train()

def main():
    parser = argparse.ArgumentParser(description="PyTorch Template MNIST Training")
    parser.add_argument(
        "--config", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config != "":
        cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    
    build_output(cfg, args.config)

    logger = setup_logger(cfg.OUTPUT_DIR)

    logger.info("Running with config:\n{}".format(cfg))
    action = input("Config Confirmed ? (Y/N)").lower().strip()
    if action == 'y':
        train(cfg)    
    else:
        shutil.rmtree(cfg.OUTPUT_DIR)
        logger.info("Training stopped")
        sys.exit(1)
    

if __name__ == '__main__':
    main()

