
import argparse
import os
import sys
from os import mkdir
import datetime
import shutil

from config.config_manager import _C as cfg
from data.build_loader import build_plain_reid_loader, build_update_reid_loader
from engine.engines.engine_reid_trick import ReIDEngine
from solver.optimizer import Solver
from visualizer.visualizer import Visualizer
from model.managers.manager_reid_ssg import SSGManager
from model.utility import get_self_label
from tools.logger import setup_logger
import torch.nn as nn
import torch


def train(cfg):

    trt_train_loader, _, _ = build_plain_reid_loader(cfg)

    manager = SSGManager(cfg)
    manager.use_multigpu()

    opts = []    
    for _loss in manager.loss_has_param:
        opts.append(Solver(cfg, _loss.named_parameters(), _lr=cfg.SOLVER.CENTER_LOSS_LR, _name="SGD", _lr_policy="none"))
    opts.append(Solver(cfg, manager.model.named_parameters()))
    visualizer = Visualizer(cfg)
    

    for cycle in range(0, cfg.REID.CYCLE):
        trt_feats = manager.extract_features(trt_train_loader, cycle)

        dists = []
        for feat in trt_feats:
            m = feat.size(0)
            distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                        torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(m, m).t()
            distmat.addmm_(1, -2, feat, feat.t())
            dists.append(distmat.numpy())
    
        labels = manager.get_self_label(dists, cycle)

        trt_update_train_loader, trt_query_loader, trt_gallery_loader = build_update_reid_loader(cfg, labels)

        cfg.SOLVER.ITERATIONS_PER_EPOCH = len(trt_update_train_loader)

        engine = ReIDEngine(cfg, opts, trt_update_train_loader, trt_query_loader, trt_gallery_loader, visualizer, manager)  
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
    
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg.OUTPUT_DIR = "{}_{}_{}".format(cfg.OUTPUT_DIR, time, cfg.EXPERIMENT)
    if cfg.OUTPUT_DIR and not os.path.exists(cfg.OUTPUT_DIR):
        mkdir(cfg.OUTPUT_DIR)
        if args.config != "":
            shutil.copy(args.config, os.path.join(cfg.OUTPUT_DIR, args.config.split("/")[-1]))

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

