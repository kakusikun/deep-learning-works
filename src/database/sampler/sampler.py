import random

from torch.utils.data import dataset, sampler
import torch.distributed as dist
from collections import defaultdict
import numpy as np
import copy

class IdBasedSampler(sampler.Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        assert self.batch_size > self.num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        _ = self._build()

    def __iter__(self):
        final_idxs = self._build()
        return iter(final_idxs)

    def __len__(self):
        return self.length

    def _build(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return final_idxs

class IdBasedDistributedSampler(sampler.Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.epoch = 0
        assert self.batch_size > self.num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        _ = self._build()

    def __iter__(self):
        final_idxs = np.array(self._build()).reshape(-1,4)
        rank = dist.get_rank()
        num_replica = dist.get_world_size()
        num_sample = len(final_idxs) // num_replica
        total_size = num_replica * num_sample
        final_idxs = final_idxs[rank:total_size:num_replica].reshape(-1).tolist()
        return iter(final_idxs)

    def __len__(self):
        return self.length
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def _build(self):
        np.random.seed(self.epoch)
        random.seed(self.epoch)
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return final_idxs

class BlancedPARSampler(sampler.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.pool = None
        self.make_pool()

        _ = self._build()

    def __iter__(self):
        indice = self._build()
        return iter(indice)

    def __len__(self):
        return self.length

    def make_pool(self):
        class_indice = []
        attrs = []
        for i, (_, label) in enumerate(self.data_source.train):
            attrs.append(label)   
            class_indice.append(i) 

        attrs = np.array(attrs)
        class_indice = np.array(class_indice)

        pool = {}
        for i, name in enumerate(self.data_source.category_names):
            pool[name] = {}
            pool[name]['p'] = class_indice[attrs[:,i]==1]
            pool[name]['n'] = class_indice[attrs[:,i]==0]
        self.pool = pool

    def _build(self):
        count = 0
        for i, name in enumerate(self.data_source.category_names):
            count += self.pool[name]['p'].shape[0]
        size = int(count / len(self.data_source.category_names))

        indice = []
        for i, name in enumerate(self.data_source.category_names):
            for cat in self.pool[name].keys():
                if cat == 'p':
                    if size > len(self.pool[name]['p']):
                        indice.extend(np.random.choice(self.pool[name][cat], size=size, replace=True).tolist())
                    else:
                        indice.extend(np.random.choice(self.pool[name][cat], size=size, replace=False).tolist())
                else:
                    if size > len(self.pool[name]['n']):
                        indice.extend(np.random.choice(self.pool[name][cat], size=size, replace=True).tolist())
                    else:
                        indice.extend(np.random.choice(self.pool[name][cat], size=size, replace=False).tolist())
        random.shuffle(indice)
        self.length = len(indice) 
        return indice
        
        
