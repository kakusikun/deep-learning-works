# encoding: utf-8
"""


"""

import os
import sys

import logging as logger
from torch.utils.tensorboard import SummaryWriter

def Visualizer(cfg):
    path = os.path.join(cfg.OUTPUT_DIR, "log")
    if not os.path.exists(path):
        os.mkdir(path)
    logger.info(path)
    visualizer = SummaryWriter(path)

    return visualizer
