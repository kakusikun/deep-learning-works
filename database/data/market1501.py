from database.data import *
import os.path as osp
import os
import re

class Market1501(BaseData):
    """
    Market1501

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html
    
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """   

    def __init__(self, cfg, **kwargs):
        self.dataset_dir = osp.join(cfg.DB.PATH, cfg.DB.DATA)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self._check_before_run()

        if cfg.DB.USE_TRAIN:
            train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
            self.train['indice'] = train
            self.train['n_samples'] = num_train_pids
            logger.info("=> {} TRAIN loaded".format(cfg.DB.DATA.upper()))
            logger.info("Dataset statistics:")
            logger.info("  ------------------------------")
            logger.info("  subset   | # ids | # images")
            logger.info("  ------------------------------")
            logger.info("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
            logger.info("  ------------------------------")

        if cfg.DB.USE_TEST:
            query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
            gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
            self.query['indice'] = query
            self.gallery['indice'] = gallery
            self.query['n_samples'] = num_query_pids
            self.gallery['n_samples'] = num_gallery_pids            
            logger.info("=> {} VAL loaded".format(cfg.DB.DATA.upper()))
            logger.info("Dataset statistics:")
            logger.info("  ------------------------------")
            logger.info("  subset   | # ids | # images")
            logger.info("  ------------------------------")
            logger.info("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
            logger.info("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
            logger.info("  ------------------------------")

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = [osp.join(root, f) for root, _, files in os.walk(dir_path) 
                               for f in files if 'jpg' in f or 'png' in f]
        pattern = re.compile(r'([-\d]+)_c(\d+)s')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            # assert 0 <= pid <= 1501  # pid == 0 means background
            # assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs