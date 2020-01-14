from database.data import *
from torchvision.datasets.cifar import CIFAR10
import os.path as osp

class Cifar10(BaseData):
    def __init__(self, cfg):
        self.dataset_dir = osp.join(cfg.DB.PATH, cfg.DB.DATA)
        self._check_before_run()
        if cfg.DB.USE_TRAIN:
            self.train['handle'] = CIFAR10(root=self.dataset_dir, train=True, transform=None, download=True)
            self.train['n_samples'] = len(self.train['handle'])
            logger.info("=> {} TRAIN loaded".format(cfg.DB.DATA.upper()))
            logger.info("Dataset statistics:")
            logger.info("  ------------------------------")
            logger.info("  subset   | # class | # images")
            logger.info("  ------------------------------")
            logger.info("  train    | {:7d} | {:8d}".format(10, self.train['n_samples']))
            logger.info("  ------------------------------")
        if cfg.DB.USE_TEST:
            self.val['handle'] = CIFAR10(root=self.dataset_dir, train=False, transform=None, download=True)
            self.val['n_samples'] = len(self.val['handle'])
            logger.info("=> {} VAL loaded".format(cfg.DB.DATA.upper()))
            logger.info("Dataset statistics:")
            logger.info("  ------------------------------")
            logger.info("  subset   | # class | # images")
            logger.info("  ------------------------------")
            logger.info("  val      | {:7d} | {:8d}".format(10, self.val['n_samples']))
            logger.info("  ------------------------------")

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))