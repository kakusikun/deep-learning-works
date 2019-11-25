import argparse
import shutil
import sys
import logging

from config.config_manager import _C as cfg
from config.config_manager import build_output
from data.build_loader import build_coco_person_loader
from engine.engines.engine_center import CenterEngine
from solver.optimizer import Solver
from visualizer.visualizer import Visualizer
from model.managers.manager_center import CenterManager
from tools.logger import setup_logger
from tools.utils import deploy_gpu
import torch.nn as nn

def train(cfg):

    train_loader, val_loader = build_coco_person_loader(cfg)

    manager = CenterManager(cfg)

    manager.use_multigpu()

    cfg.SOLVER.ITERATIONS_PER_EPOCH = len(train_loader)

    opts = []    
    opts.append(Solver(cfg, manager.model.named_parameters()))

    visualizer = Visualizer(cfg)
    
    engine = CenterEngine(cfg, opts, train_loader, val_loader, visualizer, manager)  

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

    deploy_gpu(cfg)
    train(cfg)    
    

if __name__ == '__main__':
    main()
