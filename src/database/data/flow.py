from src.database.data import *
import os.path as osp
import re
from src.database.data.market1501 import Market1501, Market1501LMDB

class FLOW(Market1501):
    """
    using market1501 data arrangement
    
    Dataset statistics:
    # identities: 971
    # images: 3884 (train) + 0 (query) + 0 (gallery)
    """
    
    def __init__(self, path="", branch="", use_train=False, use_test=False, **kwargs):
        super().__init__(path=path, branch=branch, use_train=use_train, use_test=use_test)

        
    def _process_dir(self, dir_path, relabel=False):
        img_paths = [osp.join(root, f) for root, _, files in os.walk(dir_path) 
                               for f in files if 'jpg' in f or 'png' in f]
        pattern = re.compile(r'([-\d]+)_c(\d+)s')

        pid_container = set()
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1 or camid == 4: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1 or camid == 4: continue  # junk images are just ignored
            # assert 0 <= pid <= 1501  # pid == 0 means background
            # assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs