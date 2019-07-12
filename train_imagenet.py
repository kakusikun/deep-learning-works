
import argparse
import os
import sys
from os import mkdir
import datetime
import shutil

from config.config_manager import _C as cfg
from data.build_loader import build_imagenet_loader
from engine.engines.imagenet_engine import ImageNetEngine
from solver.optimizer import Solver
from visualizer.visualizer import Visualizer
from model.managers.imagenet_manager import ImageNetManager
from model.utility import CrossEntropyLossLS
from tools.logger import setup_logger
import torch.nn as nn


def train(cfg): 

    train_loader, val_loader = build_imagenet_loader(cfg)

    model_manager = ImageNetManager(cfg)

    cfg.SOLVER.ITERATIONS_PER_EPOCH = len(train_loader)

    opts = []    
    opts.append(Solver(cfg, model_manager.model.named_parameters()))

    visualizer = Visualizer(cfg)
    
    # engine = ImageNetEngine(cfg, nn.CrossEntropyLoss(), opt, train_loader, val_loader, visualizer, model_manager)
    engine = ImageNetEngine(cfg, opts, train_loader, val_loader, visualizer, model_manager)  
    engine.Train()

def main():
    parser = argparse.ArgumentParser(description="PyTorch Template MNIST Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg.OUTPUT_DIR = "{}_{}_{}".format(cfg.OUTPUT_DIR, cfg.EXPERIMENT, time)
    if cfg.OUTPUT_DIR and not os.path.exists(cfg.OUTPUT_DIR):
        mkdir(cfg.OUTPUT_DIR)
        if args.config_file != "":
            shutil.copy(args.config_file, os.path.join(cfg.OUTPUT_DIR, args.config_file.split("/")[-1]))

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

