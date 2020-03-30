from src.database.data import *
import os
import os.path as osp

class Emotion(BaseData):
    def __init__(self, path="", use_train=False, use_test=False, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(path, 'emotion')
        self.train_dir = osp.join(self.dataset_dir, 'emotion_train_clean_112_112')
        self.val_dir = osp.join(self.dataset_dir, 'emotion_test_112_112')        
        self._check_before_run()
        if use_train:
            train, train_num_images, train_num_classes = self._process_dir(self.train_dir)
            self.train['indice'] = train
            self.train['n_samples'] = train_num_images
            logger.info("=> Emotion TRAIN loaded")
            logger.info("Dataset statistics:")
            logger.info("  ------------------------------")
            logger.info("  subset   | # class | # images")
            logger.info("  ------------------------------")
            logger.info("  train    | {:7d} | {:8d}".format(train_num_classes, self.train['n_samples']))
            logger.info("  ------------------------------")
        if use_test:
            val, val_num_images, val_num_classes = self._process_dir(self.val_dir)
            self.val['indice'] = val
            self.val['n_samples'] = val_num_images
            logger.info("=> Emotion VAL loaded")
            logger.info("Dataset statistics:")
            logger.info("  ------------------------------")
            logger.info("  subset   | # class | # images")
            logger.info("  ------------------------------")
            logger.info("  val    | {:7d} | {:8d}".format(val_num_classes, self.val['n_samples']))
            logger.info("  ------------------------------")

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
    
    def _process_dir(self, path):
        img_paths = [(osp.join(root, f)) for root, _, files in os.walk(path) for f in files if '.png' in f or '.jpg' in f]
        dataset = []
        gt = []
        for img_path in img_paths:
            label = int(img_path.split('/')[-2])
            gt.append(label)
            dataset.append((img_path, label))
        return dataset, len(img_paths), len(set(gt))