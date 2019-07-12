
import torch.utils.data as data
import lmdb
from PIL import Image
from data import caffe_pb2
datum = caffe_pb2.Datum()

class build_image_dataset(data.Dataset):
    def __init__(self, dataset, transform=None, use_lmdb=None):
        self.dataset = dataset
        self.transform = transform
        if use_lmdb is not None:  
            lmdb_env = lmdb.open(use_lmdb)
            self.lmdb_txn = lmdb_env.begin()
        else:
            self.lmdb_txn = None
    
    def __getitem__(self, index):
        img_path, label = self.dataset[index]

        if self.lmdb_txn is not None:
            raw = self.lmdb_txn.get(img_path.encode())
            datum.ParseFromString(raw)
            img = Image.frombytes('RGB', (256,256), datum.data)
            label = datum.label
        else:
            img = Image.open(img_path)        
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        return len(self.dataset)


class build_reid_dataset(data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, index):
        img_path, pids, camids = self.dataset[index]
            
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, pids, camids
    
    def __len__(self):
        return len(self.dataset)

