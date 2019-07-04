import random

from torch.utils.data import dataset, sampler
from collections import defaultdict

class IdBasedSampler(sampler.Sampler):

    @staticmethod
    def _sample(population, k):
        if len(population) < k:
            population = population * k
        return random.sample(population, k)

    def __init__(self, dataset, K=8):
        """
        :param data_source: Market1501 dataset
        :param batch_image: batch image size for one person id
        """
        super(IdBasedSampler, self).__init__(dataset)
        self.dataset = dataset.dataset
        self.id_map = defaultdict(list)
        self._build_id_map()  

        self.pids = list(self.id_map.keys())
        self.K = K
            

    def __iter__(self):
        self.indice = [] 
        random.shuffle(self.pids)        
        for pid in self.pids:
            self.indice.extend(self._sample(self.id_map[pid], self.K))
        return iter(self.indice)

    def __len__(self):
        return len(self.pids) * self.K

    def _build_id_map(self):
        for i, (_, pid, _) in enumerate(self.dataset):
            self.id_map[pid].append(i)


