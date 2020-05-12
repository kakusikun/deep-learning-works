import os
import argparse
import shutil
import logging
import time
import sys
import traceback
import torch
import torch.distributed as dist

from src.factory.config_factory import cfg
from src.factory.config_factory import build_output, show_products, show_configs
from src.factory.trainer_factory import TrainerFactory
from tools.utils import deploy_macro
from tools.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="PyTorch Deep Learning")
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--config", default="", help="path to config file", type=str)
    parser.add_argument('--products', action='store_true',
                        help='list available products in all factories')
    parser.add_argument('--cfg', action='store_true',
                        help='list available configs in YAML')
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.products:
        show_products()

    if args.cfg:
        show_configs()

    if args.config != "":
        cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)    
    if args.local_rank != 0:
        time.sleep(5)
        cfg.IO = False
        cfg.SAVE = False
    build_output(cfg, args.config)
    logger = setup_logger(cfg.OUTPUT_DIR)   

    deploy_macro(cfg)

    assert cfg.DISTRIBUTED is True
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    logger.info(f"Rank [{rank}] Start!")
    device = torch.device("cuda:{}".format(args.local_rank))
    torch.cuda.set_device(device)
    trainer = TrainerFactory.produce(cfg)

    logger.info("Running with config")
    
    if cfg.EVALUATE:
        trainer.test()
        sys.exit()
    
    try:
        trainer.train()        
    except:
        logger.info("Unexpected Error Occurred")
        if cfg.SAVE:
            logger.info("Back up the Checkpoint")
            trainer.graph.save(trainer.graph.save_path, trainer.graph.model, trainer.graph.sub_models, trainer.solvers, trainer.engine.epoch, trainer.engine.accu)
        logger.info(traceback.format_exc())
        sys.exit(1)
    

if __name__ == '__main__':
    # python3 -m torch.distributed.launch --nproc_per_node=3
    # kill $(ps aux | grep train_distributed.py | grep -v grep | awk '{print $2}')
    print(os.getpid())
    main()