import argparse
import shutil
import sys
import logging

from config.config_factory import _C as cfg
from config.config_factory import build_output
from database.loader_factory import get_loader
from engine.engine_factory import get_engine
from manager.manager_factory import get_manager

from solver.optimizer import Solver
from visualizer.visualizer import Visualizer
from tools.logger import setup_logger
from tools.utils import deploy_macro
import torch.nn as nn

def train(cfg):
    loader = get_loader(cfg.DB.LOADER)(cfg)
    manager = get_manager(cfg.MANAGER)(cfg)
    manager.use_multigpu()

    opts = []    
    opts.append(Solver(cfg, manager.model.named_parameters()))
    visualizer = Visualizer(cfg)    
    engine = get_engine(cfg.ENGINE)(cfg, opts, loader, visualizer, manager)  

    logger = logging.getLogger("logger")
    logger.info("Running with config:\n{}".format(cfg))
    action = input("Config Confirmed ? (Y/N)").lower().strip()
    if action == 'y':
        if cfg.EVALUATE:
            engine.Evaluate()
            sys.exit(1)
        engine.Train()
    else:
        shutil.rmtree(cfg.OUTPUT_DIR)
        logger.info("Training stopped")
        sys.exit(1)

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

    deploy_macro(cfg)
    train(cfg)    
    

if __name__ == '__main__':
    main()

