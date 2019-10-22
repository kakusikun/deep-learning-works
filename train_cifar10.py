import argparse
import shutil
import sys

from config.config_manager import _C as cfg
from config.config_manager import build_output
from data.build_loader import build_cifar10_loader
from engine.engine_imagenet import ImageNetEngine
from solver.optimizer import Solver
from visualizer.visualizer import Visualizer
from model.manager import TrainingManager
from model.utility import CrossEntropyLossLS
import logging
logger = logging.getLogger("logger")

import torch.nn as nn


def train(cfg):

    train_loader, val_loader = build_cifar10_loader(cfg)

    manager = TrainingManager(cfg)

    manager.use_multigpu()

    if cfg.EVALUATE:
        engine = ImageNetEngine(cfg, None, None, None, val_loader, None, manager)
        engine.Inference()
        sys.exit(1)

    cfg.SOLVER.ITERATIONS_PER_EPOCH = len(train_loader)

    opt = Solver(cfg, manager.params)

    visualizer = Visualizer(cfg)
    
    engine = ImageNetEngine(cfg, CrossEntropyLossLS(cfg.MODEL.NUM_CLASSES), opt, train_loader, val_loader, visualizer, manager)
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

