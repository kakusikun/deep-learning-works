from src.database.data import *
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

    def __init__(self, path="", branch="", use_train=False, use_test=False, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(path, branch)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self._check_before_run()

        if use_train:
            train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
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

class Market1501LMDB(BaseData):
    """
    Market1501

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html
    
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    Source = Market1501

    def __init__(self, path="", branch="", use_train=False, use_test=False, **kwargs):
        super().__init__()
        self.data_dir =  osp.join(path, branch)
        self.lmdb_dir = osp.join(self.data_dir, 'lmdb')
        self.train_list = osp.join(self.data_dir, 'bounding_box_train.txt')
        self.query_list = osp.join(self.data_dir, 'query.txt')
        self.gallery_list = osp.join(self.data_dir, 'bounding_box_test.txt')
        if not osp.exists(self.lmdb_dir):
            logger.info("LMDB does not exist, prepare to make one ...")
            _data = self.Source(path=path, branch=branch.split('_')[0], use_train=use_train, use_test=use_test)
            _data.make_lmdb(self.data_dir)
        env = lmdb.open(self.lmdb_dir)
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
            logger.info(f"  gallery  | {num_gallery_pids:7d} | {num_gallery_imgs:8d}")            
            logger.info("  ------------------------------")
        
    def _process_list(self, list_path, relabel=False):
        dataset = []
        pids = set()
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
                    dataset.append((img_path.encode(), pid2label[pid], int(cid)))
        else:
            with open(list_path, 'r') as f:
                for line in f.readlines():
                    img_path, pid, cid = line.strip().split(",")
                    pids.add(pid)
                    dataset.append((img_path.encode(), int(pid), int(cid)))

        num_pids = len(pids)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs