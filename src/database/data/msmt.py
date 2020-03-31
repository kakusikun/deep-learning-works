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

    def __init__(self, path="", branch="", use_train=False, use_test=False, use_all=False, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(path, branch)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self._check_before_run()

        if use_all:
            train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
            extra, num_extra_pids, num_extra_imgs = self._process_dir([self.query_dir, self.gallery_dir], relabel=True, offset=num_train_pids)
            train.extend(extra)
            num_train_pids += num_extra_pids
            num_train_imgs += num_extra_imgs
            self.train['indice'] = train
            self.train['n_samples'] = num_train_pids
            logger.info("=> {} TRAIN loaded".format(branch.upper()))
            logger.info("Dataset statistics:")
            logger.info("  ------------------------------")
            logger.info("  subset   | # ids | # images")
            logger.info("  ------------------------------")
            logger.info(f"  train    | {num_train_pids:7d} | {num_train_imgs:8d}")            
            logger.info("  ------------------------------")
        else:
            if use_train:
                train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
                self.train['indice'] = train
                self.train['n_samples'] = num_train_pids
                logger.info("=> {} TRAIN loaded".format(branch.upper()))
                logger.info("Dataset statistics:")
                logger.info("  ------------------------------")
                logger.info("  subset   | # ids | # images")
                logger.info("  ------------------------------")
                logger.info(f"  train    | {num_train_pids:7d} | {num_train_imgs:8d}")            
                logger.info("  ------------------------------")

            if use_test:
                query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
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
                logger.info(f"  query    | {num_query_pids:7d} | {num_query_imgs:8d}")            
                logger.info(f"  gallery    | {num_gallery_pids:7d} | {num_gallery_imgs:8d}")            
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

    def _process_dir(self, dir_path, relabel=False, offset=0):
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
        pid2label = {pid:label+offset for label, pid in enumerate(pid_container)}

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
    
    def make_lmdb(self, path):
        lmdb_path = osp.join(path, 'lmdb')
        train_list = osp.join(path, 'bounding_box_train.txt')
        query_list = osp.join(path, 'query.txt')
        gallery_list = osp.join(path, 'bounding_box_test.txt')
        if not osp.exists(path):
            os.mkdir(path)
            os.mkdir(lmdb_path)

        lmdb_env = lmdb.open(lmdb_path, map_size=int(1e12))
        lmdb_txn = lmdb_env.begin(write=True)
        for (img_path, _, _) in tqdm(self.train['indice']):
            f_img = open(img_path, 'rb')
            img_str = f_img.read()
            key = img_path.split(self.dataset_dir+"/")[-1]
            lmdb_txn.put(key.encode(), img_str)
        for (img_path, _, _) in tqdm(self.query['indice']):
            f_img = open(img_path, 'rb')
            img_str = f_img.read()
            key = img_path.split(self.dataset_dir+"/")[-1]
            lmdb_txn.put(key.encode(), img_str)
        for (img_path, _, _) in tqdm(self.gallery['indice']):
            f_img = open(img_path, 'rb')
            img_str = f_img.read()
            key = img_path.split(self.dataset_dir+"/")[-1]
            lmdb_txn.put(key.encode(), img_str)
        lmdb_txn.commit()

        with open(train_list, 'w') as f:
            for (img_path, pid, cid) in tqdm(self.train['indice']):
                key = img_path.split(self.dataset_dir+"/")[-1]
                f.write(f"{key},{pid},{cid}\n")
        with open(query_list, 'w') as f:
            for (img_path, pid, cid) in tqdm(self.query['indice']):
                key = img_path.split(self.dataset_dir+"/")[-1]
                f.write(f"{key},{pid},{cid}\n")
        with open(gallery_list, 'w') as f:
            for (img_path, pid, cid) in tqdm(self.gallery['indice']):
                key = img_path.split(self.dataset_dir+"/")[-1]
                f.write(f"{key},{pid},{cid}\n")
        
    
class MSMT17LMDB(BaseData):
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

    def __init__(self, path="", branch="", use_train=False, use_test=False, use_all=False, **kwargs):
        super().__init__()
        self.data_dir =  osp.join(path, branch)
        self.lmdb_dir = osp.join(self.data_dir, 'lmdb')
        self.train_list = osp.join(self.data_dir, 'bounding_box_train.txt')
        self.query_list = osp.join(self.data_dir, 'query.txt')
        self.gallery_list = osp.join(self.data_dir, 'bounding_box_test.txt')
        if not osp.exists(self.lmdb_dir):
            logger.info("LMDB does not exist, prepare to make one ...")
            _data = MSMT17(path=path, branch=branch.split('_')[0], use_train=use_train, use_test=use_test)
            _data.make_lmdb(self.data_dir)
        env = lmdb.open(self.lmdb_dir)
        if use_all:
            train, num_train_pids, num_train_imgs = self._process_list(self.train_list, relabel=True)
            extra, num_extra_pids, num_extra_imgs = self._process_list([self.query_list, self.gallery_list], relabel=True, offset=num_train_pids)
            train.extend(extra)
            num_train_pids += num_extra_pids
            num_train_imgs += num_extra_imgs
            self.train['indice'] = train
            self.train['n_samples'] = num_train_pids
            logger.info("=> {} TRAIN loaded".format(branch.upper()))
            logger.info("Dataset statistics:")
            logger.info("  ------------------------------")
            logger.info("  subset   | # ids | # images")
            logger.info("  ------------------------------")
            logger.info(f"  train    | {num_train_pids:7d} | {num_train_imgs:8d}")            
            logger.info("  ------------------------------")
        else:
            if use_train:
                train, num_train_pids, num_train_imgs = self._process_list(self.train_list, relabel=True)
                self.train['indice'] = train
                self.train['n_samples'] = num_train_pids
                self.train['handle'] = env.begin()
                logger.info("=> {} TRAIN loaded".format(branch.upper()))
                logger.info("Dataset statistics:")
                logger.info("  ------------------------------")
                logger.info("  subset   | # ids | # images")
                logger.info("  ------------------------------")
                logger.info(f"  train    | {num_train_pids:7d} | {num_train_imgs:8d}")            
                logger.info("  ------------------------------")

            if use_test:
                query, num_query_pids, num_query_imgs = self._process_list(self.query_list, relabel=False)
                gallery, num_gallery_pids, num_gallery_imgs = self._process_list(self.gallery_list, relabel=False)
                self.query['handle'] = env.begin()
                self.query['indice'] = query
                self.query['n_samples'] = num_query_pids
                self.gallery['handle'] = env.begin()
                self.gallery['indice'] = gallery
                self.gallery['n_samples'] = num_gallery_pids
                logger.info("=> {} VAL loaded".format(branch.upper()))
                logger.info("Dataset statistics:")
                logger.info("  ------------------------------")
                logger.info("  subset   | # ids | # images")
                logger.info("  ------------------------------")
                logger.info(f"  query    | {num_query_pids:7d} | {num_query_imgs:8d}")            
                logger.info(f"  gallery    | {num_gallery_pids:7d} | {num_gallery_imgs:8d}")            
                logger.info("  ------------------------------")
        
    def _process_list(self, list_paths, relabel=False, offset=0):
        dataset = []
        pids = set()
        if not isinstance(list_paths, list):
            list_paths = [list_paths]
        for list_path in list_paths:
            if relabel:
                with open(list_path, 'r') as f:
                    for line in f.readlines():
                        _, pid, _ = line.strip().split(",")
                        pids.add(pid)
                pid2label = {pid:label for label, pid in enumerate(pids)}
                with open(list_path, 'r') as f:
                    for line in f.readlines():
                        img_path, pid, cid = line.strip().split(",")
                        pids.add(pid)
                        dataset.append((img_path.encode(), pid2label[pid] + offset, int(cid)))
            else:
                with open(list_path, 'r') as f:
                    for line in f.readlines():
                        img_path, pid, cid = line.strip().split(",")
                        pids.add(pid)
                        dataset.append((img_path.encode(), int(pid) + offset, int(cid)))

        num_pids = len(pids)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs