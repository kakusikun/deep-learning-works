import argparse
import shutil
import logging
import sys

from src.factory.config_factory import cfg
from src.factory.config_factory import build_output, show_products, show_configs
from src.factory.trainer_factory import TrainerFactory
from tools.utils import deploy_macro
from tools.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="PyTorch Deep Learning")
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
    build_output(cfg, args.config)
    logger = setup_logger(cfg.OUTPUT_DIR)
    deploy_macro(cfg)
    trainer = TrainerFactory.produce(cfg)

    logger.info("Running with config")
    
    if cfg.EVALUATE:
        trainer.test()
        sys.exit(1)
    trainer.train()        
    

if __name__ == '__main__':
    main()