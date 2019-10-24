import argparse
import shutil
import os
import sys
from glob import glob

from config.config_manager import _C as cfg
from config.config_manager import build_output
from data.build_loader import build_plain_reid_loader, build_update_reid_loader
from engine.engines.engine_reid_ssg import SSGEngine
from solver.optimizer import Solver
from visualizer.visualizer import Visualizer
from model.managers.manager_reid_ssg import SSGManager
from model.utility import get_self_label

from tools.logger import setup_logger
import torch.nn as nn
import torch


def train(cfg):

    trt_train_loader, _, _ = build_plain_reid_loader(cfg)

    for cycle in range(0, cfg.REID.CYCLE):
        manager = SSGManager(cfg)
        manager.use_multigpu()   
        manager.set_save_path("cycle_{}".format(cycle))
        trt_feats = manager.extract_features(trt_train_loader, cycle)
        dists = manager.get_feature_dist(trt_feats)    
        labels = manager.get_self_label(dists, cycle)
        #  import numpy as np
        #  labels  =  np.random.randint(0,702,len(trt_train_loader.dataset)).tolist()

        trt_update_train_loader, trt_query_loader, trt_gallery_loader = build_update_reid_loader(cfg, labels)

        cfg.SOLVER.ITERATIONS_PER_EPOCH = len(trt_update_train_loader)

        opts = [Solver(cfg, manager.model.named_parameters())]
        visualizer = Visualizer(cfg, "cycle_{}/log".format(cycle))

        engine = SSGEngine(cfg, opts, trt_update_train_loader, trt_query_loader, trt_gallery_loader, visualizer, manager)  
        engine.Train()
        cfg.RESUME = os.path.join(os.getcwd(), sorted(glob(manager.save_path + "/model*"))[-1])

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

