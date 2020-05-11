from src.database.data import *
import os.path as osp

# find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
class ImageNet(BaseData):
    def __init__(self, path="", branch="", use_train=False, use_test=False, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(path, branch)
        self.train_dir = osp.join(self.dataset_dir, "ilsvrc2012_train")
        self.val_dir = osp.join(self.dataset_dir, "ilsvrc2012_val")
        self.train_list = osp.join(self.dataset_dir, "ilsvrc2012_train.txt")
        self.val_list = osp.join(self.dataset_dir, "ilsvrc2012_val.txt")
        self.class_dict = {}
        self._check_before_run()
        if use_train:
            train, train_num_images, train_num_classes = self._process_train_dir()
            self.train['handle'] = lmdb.open(self.train_dir)
            self.train['indice'] = train
            self.train['n_samples'] = train_num_images
            logger.info("=> {} TRAIN loaded".format(branch.upper()))
            logger.info("Dataset statistics:")
            logger.info("  ------------------------------")
            logger.info("  subset   | # class | # images")
            logger.info("  ------------------------------")
            logger.info("  train    | {:7d} | {:8d}".format(train_num_classes, train_num_images))
            logger.info("  ------------------------------")
        if use_test:
            val, val_num_images, val_num_classes = self._process_val_dir()
            self.val['handle'] = lmdb.open(self.val_dir)
            self.val['indice'] = val
            self.val['n_samples'] = val_num_images
            logger.info("=> {} VAL loaded".format(branch.upper()))
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

def make_lmdb(src, train=False):
    img_src = osp.join(src, "ILSVRC2012_img_train") if train else osp.join(src, "ILSVRC2012_img_val")
    lmdb_path = osp.join(src, 'ilsvrc2012_train') if train else osp.join(src, 'ilsvrc2012_val')
    if not os.path.exists(lmdb_path):
        os.mkdir(lmdb_path)
    img_list_path = osp.join(src, "ilsvrc2012_train.txt") if train else osp.join(src, "ilsvrc2012_val.txt")

    lmdb_env = lmdb.open(
        lmdb_path, 
        map_size = 1099511627776 * 2,
        readonly = False,
        meminit = False,
        map_async = True
    )
    lmdb_txn = lmdb_env.begin(write=True)

    with open(img_list_path, 'r') as f:
        for line in tqdm(f):
            img_name, label = line.strip().split(" ")
            fname = osp.join(img_src, img_name)        
            f_img = open(fname, 'rb')
            img_str = f_img.read()
            lmdb_txn.put(img_name.encode(), img_str)
    lmdb_txn.commit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Deep Learning")
    parser.add_argument("--src", default="", help="directory having ILSVRC2012_img_train or ILSVRC2012_img_val", type=str)
    parser.add_argument('--train', action='store_true', help='for training data')
    args = parser.parse_args()

    make_lmdb(args.src, args.train)