from src.database.data import *
import os.path as osp
import glob
import re
from collections import defaultdict

class MSMT17(BaseData):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html
    
    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    dataset_dir = 'msmt17'

    def __init__(self, path="", branch="", use_train=False, use_test=False, is_merge=False **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(path, branch)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self._check_before_run()

        if use_train:
            train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
            if is_merge:
                train, num_train_pids, num_train_imgs = self._process_dir([self.train_dir, self.query_dir], relabel=True)
                train, num_train_pids, num_train_imgs = self.clean_dataset(train, relabel=True)
            self.train['indice'] = train
            self.train['n_samples'] = num_train_pids
            logger.info("=> {} TRAIN loaded".format(branch.upper()))
            logger.info("Dataset statistics:")
            logger.info("  ------------------------------")
            logger.info("  subset   | # ids | # images")
            logger.info("  ------------------------------")
            logger.info("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
            logger.info("  ------------------------------")

        if use_test:
            query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
            if is_merge:
                query, num_query_pids, num_query_imgs = self.clean_dataset(query, method='gt')
            gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
            self.query['indice'] = query
            self.gallery['indice'] = gallery
            self.query['n_samples'] = num_query_pids
            self.gallery['n_samples'] = num_gallery_pids            
            logger.info("=> {} VAL loaded".format(branch.upper()))
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
        if isinstance(dir_path, list):
            img_paths = [] 
            for _dir_path in dir_path:
                img_paths.extend(glob.glob(osp.join(_dir_path, '*.jpg'))) 
        else:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

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
            camid -= 1 # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs
    
    def clean_dataset(self, dataset, method='lt', relabel=False):
        count = defaultdict(int)
        for _, pid, _ in dataset:
            count[pid] += 1 
            
        delete_pids = []
        pid_container = set()
        for pid in count.keys():
            if method == 'lt' and count[pid] < 4:
                delete_pids.append(pid)
            elif method == 'gt' and count[pid] >= 4: 
                delete_pids.append(pid)
            else:
                pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        new_dataset = []
        for img_path, pid, camid in dataset:
            if pid not in delete_pids:
                if relabel: pid = pid2label[pid]
                new_dataset.append((img_path, pid, camid))
        return new_dataset, len(count) - len(delete_pids), len(new_dataset)