import argparse
import shutil
import logging
import sys
import numpy as np

from config.config_factory import _C as cfg
from config.config_factory import build_output
from trainer.trainer_factory import get_trainer

from tools.logger import setup_logger
from tools.utils import deploy_macro, print_config

from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler

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
    deploy_macro(cfg)    

    def trial_str_creator(trial):
        return f"{trial.trainable_name}_{trial.trial_id}"

    def train_with_tune(config):
        build_output(cfg, args.config)
        logger = setup_logger(cfg.OUTPUT_DIR)
        logger.info(cfg.OUTPUT_DIR)
        cfg.SOLVER.MOMENTUM = config['momentum']
        cfg.SOLVER.BASE_LR = config['lr']
        cfg.SOLVER.WARMRESTART_PERIOD = config['restart_period']
        trainer = get_trainer(cfg.TRAINER)(cfg)
        trainer.train()
        acc = trainer.acc
        track.log(mean_accuracy=acc)

    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", metric="mean_accuracy")  

    config = {
        "lr": tune.sample_from(lambda spec: 10**(-3 * np.random.rand())),
        "momentum": tune.uniform(0.1, 0.9),
        "restart_period": tune.randint(10,30)}

    analysis = tune.run(
        train_with_tune,
        trial_name_creator=trial_str_creator,
        name=cfg.EXPERIMENT,
        scheduler=sched,
        stop={
            "mean_accuracy": 0.90},
        resources_per_trial={
            "cpu": 2,
            "gpu": 1},
        num_samples=4,
        config=config)

    print(f'Best config is: {analysis.get_best_config(metric="mean_accuracy")}')

if __name__ == '__main__':
    main()