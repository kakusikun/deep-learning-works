from src.database.data import *
import os
import os.path as osp
import re
from collections import defaultdict

class TinyImageNet(BaseData):
    def __init__(self, path="", branch="", use_train=False, use_test=False, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(path, branch)
        self.train_dir = osp.join(self.dataset_dir, "train")
        self.val_dir = osp.join(self.dataset_dir, "val")
        self.val_list = osp.join(self.val_dir, "val_annotations.txt")
        self.class_list = osp.join(self.dataset_dir, "wnids.txt")
        self.class_dict = {}
        self._build_class_dict()
        self._check_before_run()
        if use_train:
            train, train_num_images, train_stats = self._process_train_dir()
            self.train['indice'] = train
            self.train['n_samples'] = train_num_images
            logger.info("=> {} TRAIN loaded".format(branch.upper()))
            logger.info("Dataset statistics:")
            logger.info("  ------------------------------")
            logger.info("  subset   | # class | # images")
            logger.info("  ------------------------------")
            for class_index in train_stats:
                logger.info(f"  train    | {class_index:7d} | {train_stats[class_index]:8d}")
            logger.info(f"  train    | {len(train_stats):7d} | {train_num_images:8d}")            
            logger.info("  ------------------------------")
        if use_test:
            val, val_num_images, val_stats = self._process_val_dir()
            self.val['indice'] = val
            self.val['n_samples'] = val_num_images
            logger.info("=> {} VAL loaded".format(branch.upper()))
            logger.info("Dataset statistics:")
            logger.info("  ------------------------------")
            logger.info("  subset   | # class | # images")
            logger.info("  ------------------------------")
            for class_index in val_stats:
                logger.info(f"  val    | {class_index:7d} | {val_stats[class_index]:8d}")
            logger.info(f"  val    | {len(val_stats):7d} | {val_num_images:8d}")            
            logger.info("  ------------------------------")

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.val_list):
            raise RuntimeError("'{}' is not available".format(self.val_list))
        if not osp.exists(self.class_list):
            raise RuntimeError("'{}' is not available".format(self.class_list))
            
    def _build_class_dict(self):
        with open(self.class_list) as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip().split(" ")
                self.class_dict[line[0]] = i

    def _process_train_dir(self):
        dataset = []
        stats = defaultdict(int)
        img_paths = [osp.join(root, f) for root, _, files in os.walk(self.train_dir) for f in files if '.JPEG' in f]
        pattern = re.compile(r'/train/(\w+)')
        for path in img_paths:
            cls_text = pattern.search(path).groups()[0]
            dataset.append((path, self.class_dict[cls_text]))
            stats[self.class_dict[cls_text]] += 1

        return dataset, len(dataset), stats

    def _process_val_dir(self):        
        dataset = []
        img2class = {}
        img_paths = [osp.join(root, f) for root, _, files in os.walk(self.val_dir) for f in files if '.JPEG' in f]
        stats = defaultdict(int)
        with open(self.val_list) as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                img_name, cls_text = line[:2]
                img2class[img_name] = cls_text
        
        for path in img_paths:
            dataset.append((path, self.class_dict[img2class[osp.basename(path)]]))
            stats[self.class_dict[img2class[osp.basename(path)]]] += 1
        return dataset, len(dataset), stats