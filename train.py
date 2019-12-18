import argparse
import shutil
import logging
import sys

from config.config_factory import _C as cfg
from config.config_factory import build_output
from trainer.trainer_factory import get_trainer

from tools.logger import setup_logger
from tools.utils import deploy_macro


def main():
    parser = argparse.ArgumentParser(description="PyTorch Deep Learning")
    parser.add_argument("--config", default="", help="path to config file", type=str)
    parser.add_argument('--list', action='store_true',
                        help='list available config in factories')
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.list:
        from database.data_factory import get_names as data_names
        from database.dataset_factory import get_names as dataset_names
        from database.loader_factory import get_names as loader_names
        from manager.manager_factory import get_names as manager_names
        from engine.engine_factory import get_names as engine_names
        print("DATA: ", data_names())
        print("DATASET: ", dataset_names())
        print("LOADER: ", loader_names())
        print("MANAGER: ", manager_names())
        print("ENGINE: ", engine_names())
        sys.exit(1)

    if args.config != "":
        cfg.merge_from_file(args.config)

    cfg.merge_from_list(args.opts)
    
    build_output(cfg, args.config)

    logger = setup_logger(cfg.OUTPUT_DIR)

    deploy_macro(cfg)

    trainer = get_trainer(cfg.TRAINER)(cfg)

    logger.info("Running with config:\n{}".format(cfg))
    
    if cfg.EVALUATE:
        trainer.test()
        sys.exit(1)
    trainer.train()        
    

if __name__ == '__main__':
    main()