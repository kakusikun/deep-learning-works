from database.data import *
import os.path as osp

class ImageNet(BaseData):
    def __init__(self, cfg):
        self.dataset_dir = osp.join(cfg.DB.PATH, cfg.DB.DATA)
        self.train_dir = osp.join(self.dataset_dir, "ilsvrc2012_train")
        self.val_dir = osp.join(self.dataset_dir, "ilsvrc2012_val")
        self.train_list = osp.join(self.dataset_dir, "ilsvrc2012_train.txt")
        self.val_list = osp.join(self.dataset_dir, "ilsvrc2012_val.txt")
        self.train_lmdb = osp.join(self.dataset_dir, "imagenet_256x256_lmdb_train")
        self.val_lmdb = osp.join(self.dataset_dir, "imagenet_256x256_lmdb_val")
        self.use_lmdb = False
        self.class_dict = {}
        self._check_before_run()
        if cfg.DB.USE_TRAIN:
            train, train_num_images, train_num_classes = self._process_train_dir()
            self.train['indice'] = train
            self.train['n_samples'] = train_num_images
            logger.info("=> {} TRAIN loaded".format(cfg.DB.DATA.upper()))
            logger.info("Dataset statistics:")
            logger.info("  ------------------------------")
            logger.info("  subset   | # class | # images")
            logger.info("  ------------------------------")
            logger.info("  train    | {:7d} | {:8d}".format(train_num_classes, train_num_images))
            logger.info("  ------------------------------")
        if cfg.DB.USE_TEST:
            val, val_num_images, val_num_classes = self._process_val_dir()
            self.val['indice'] = val
            self.val['n_samples'] = val_num_images
            logger.info("=> {} VAL loaded".format(cfg.DB.DATA.upper()))
            logger.info("Dataset statistics:")
            logger.info("  ------------------------------")
            logger.info("  subset   | # class | # images")
            logger.info("  ------------------------------")
            logger.info("  val      | {:7d} | {:8d}".format(val_num_classes, val_num_images))
            logger.info("  ------------------------------")

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.train_list):
            raise RuntimeError("'{}' is not available".format(self.train_list))
        if not osp.exists(self.val_list):
            raise RuntimeError("'{}' is not available".format(self.val_list))
        if osp.exists(self.train_lmdb) and osp.exists(self.val_lmdb):
            self.use_lmdb = True
            logger.info("Training LMDB is used: {}".format(self.train_lmdb))
            logger.info("Validation LMDB is used: {}".format(self.val_lmdb))
        else:
            self.train_lmdb = None
            self.val_lmdb = None
            

    def _process_train_dir(self):
        dataset = []
        with open(self.train_list, 'r') as f:
            for line in f:
                img, label = line.strip().split(" ")
                if not self.use_lmdb:
                    dataset.append((osp.join(self.train_dir, img), int(label)))                    
                else:
                    dataset.append((img, int(label)))

                class_name = img.split("/")[0]
                if class_name not in self.class_dict:
                    self.class_dict[class_name] = int(label)
                
        return dataset, len(dataset), len(self.class_dict)

    def _process_val_dir(self):        
        dataset = []
        gt = []
        with open(self.val_list, 'r') as f:
            for line in f:
                img, label = line.strip().split(" ")
                if not self.use_lmdb:
                    dataset.append((osp.join(self.val_dir, img), int(label)))
                else:
                    dataset.append((img, int(label)))
                gt.append(int(label))
        
        return dataset, len(dataset), len(set(gt))