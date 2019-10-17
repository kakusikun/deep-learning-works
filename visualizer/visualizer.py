# encoding: utf-8
"""


"""

import os
import sys

import logging as logger
from tensorboardX import SummaryWriter

def Visualizer(cfg, log_name='log'):
    path = os.path.join(cfg.OUTPUT_DIR, log_name)
    if not os.path.exists(path):
        os.mkdir(path)
    logger.info(path)
    visualizer = SummaryWriter(path)

    return visualizer
