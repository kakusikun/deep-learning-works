import argparse
import shutil
import sys
import logging

from config.config_factory import _C as cfg
from config.config_factory import build_output
from data.build_loader import build_reid_loader
from engine.engines.engine_reid_trick import ReIDEngine
from solver.optimizer import Solver
from visualizer.visualizer import Visualizer
from manager.base_managers.manager_reid_trick import TrickManager
from tools.logger import setup_logger
from tools.utils import deploy_macro
import torch.nn as nn

def train(cfg):

    train_loader, query_loader, gallery_loader = build_reid_loader(cfg)

    manager = TrickManager(cfg)

    manager.use_multigpu()

    cfg.SOLVER.ITERATIONS_PER_EPOCH = len(train_loader)

    opts = []    
    for _loss in manager.loss_has_param:
        opts.append(Solver(cfg, _loss.named_parameters(), _lr=cfg.SOLVER.CENTER_LOSS_LR, _name="SGD", _lr_policy="none"))
    opts.append(Solver(cfg, manager.model.named_parameters()))

    visualizer = Visualizer(cfg)
    
    engine = ReIDEngine(cfg, opts, train_loader, query_loader, gallery_loader, visualizer, manager)  

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

