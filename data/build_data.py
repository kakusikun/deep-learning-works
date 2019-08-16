
import torch
import os
import cv2
import numpy as np
import torch.utils.data as data
import lmdb
from PIL import Image
from data import caffe_pb2
from data.build_transform import RandomErasing, _RandomCrop, _RandomHorizontalFlip
import torchvision.transforms as T
import torchvision.transforms.functional as F
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
        img_path, pid, camid = self.dataset[index]

        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, pid, camid
    
    def __len__(self):
        return len(self.dataset)

class build_reid_atmap_dataset(data.Dataset):
    def __init__(self, dataset, cfg):
        self.dataset = dataset
        self.transform = transform
        self.at_maps = cfg.DATASET.ATTENTION_MAPS
        self.at_maps_keys = {}
        with open(cfg.DATASET.ATTENTION_MAPS_LIST, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                self.at_maps_keys[line] = i
        
        self.resize = T.Resize(size=cfg.INPUT.IMAGE_SIZE)
        self.at_map_resize = T.Resize(size=cfg.INPUT.IMAGE_SIZE, interpolation=Image.NEAREST)
        self.normalize = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        self.random_hflip = _RandomHorizontalFlip(p=cfg.INPUT.PROB)
        self.random_erase = RandomErasing()
        self.random_crop = _RandomCrop(size=cfg.INPUT.IMAGE_CROP_SIZE, padding=cfg.INPUT.IMAGE_PAD)
           
    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        at_maps_key = img_path.split("/")[-1]
        if  at_maps_key in self.at_maps_keys:
            at_map = Image.open(os.path.join(self.at_maps, at_maps_key))
            at_map_label = 1
        else:
            at_map = torch.ones(16*8)
            at_map_label = -1       
        
        img = Image.open(img_path)
        img = self.resize(img)
        img, is_flip = self.random_hflip(img)
        img, i, j, h, w = self.random_crop(img)
        img = F.to_tensor(img)
        img = self.normalize(img)
        img, x1, y1, rh, rw = self.random_erase(img)
        
        if at_map_label > 0:
            at_map = self.at_map_resize(at_map)
            if is_flip:
                at_map = F.hflip(at_map)
            at_map = self.random_crop.by_param(at_map, i, j, h, w)
            at_map = torch.Tensor(np.array(at_map)).transpose(0,2).transpose(1,2)
            if x1 > 0:
                at_map = self.random_erase.by_param(at_map, x1, y1, rh, rw)
            at_map = at_map.transpose(0,2).transpose(0,1).numpy()
            at_map = cv2.resize(at_map, (8, 16), interpolation=cv2.INTER_NEAREST).mean(axis=2)
            at_map = torch.Tensor(at_map).view(-1)    
        
        return img, pid, camid, at_map, at_map_label
    
    def __len__(self):
        return len(self.dataset)

class CALTECH(data.Dataset):
    def __init__(self, imgSrc, annoSrc, r=2, downSize=4, scale='h', offset=True, transform=None):
        self.imgSrc = imgSrc
        self.annoSrc = annoSrc
        self.transform = transform
        self.r = r
        self.down = downSize
        self.offset = offset
        self.scale = scale

        self.hashTable = defaultdict(dict)
        self.keys = []

        self.set_keys()
    
    def set_keys(self):
        rows, cols = 480, 640
        
        bboxCount = 0

        annoPaths = sorted([f for root, _, files in os.walk(self.annoSrc)
                        for f in files if 'txt' in f])

        for annoPath in tqdm(annoPaths, desc="Making List"):
            anno = annoPath.split("_")
            setIdx, videoIdx, imgIdx = anno[0], anno[1], anno[2].split(".")[0]+".jpg"
            imgPath = os.path.join(self.imgSrc, setIdx, videoIdx, "images", imgIdx)
            self.keys.append(imgPath)

            annoPath = os.path.join(self.annoSrc, annoPath)
            with open(annoPath, 'r') as f:
                igs = []
                gts = []
                for line in f:                    
                    line = line.strip().split(" ")
                    if line[0] == "%":
                        continue
                    label, x1, y1 = line[0], max(int(float(line[1])), 0), max(int(float(line[2])), 0)
                    w, h = min(int(float(line[3])), cols - x1 - 1), min(int(float(line[4])), rows - y1 - 1)
                    occlu, ignore = line[5], line[10]
                    
                    if ignore == '1':
                        igs.append([x1, y1, x1+w, y1+h])
                    else:
                        gts.append([x1, y1, x1+w, y1+h])

                    bboxCount += 1       

                self.hashTable[imgPath]['ignoreareas'] = igs
                self.hashTable[imgPath]['bboxes'] = gts

        glog.info("Total {} samples, {} boxes".format(len(self.keys), bboxCount))

    
    def __getitem__(self, index):
        imgPath = self.keys[index]
        img = Image.open(imgPath)
        w, h = img.size
        
        if self.transform is not None:
            img = self.transform(img)

        gts = np.array(self.hashTable[imgPath]['bboxes'])
        igs = np.array(self.hashTable[imgPath]['ignoreareas'])

        scaleMap = torch.zeros(2, int(h/self.down), int(w/self.down))
        if self.scale == 'hw':
           scaleMap = torch.zeros(3, int(h/self.down), int(w/self.down))

        if self.offset:
            offsetMap = torch.zeros(3, int(h/self.down), int(w/self.down))          

        semanMap = torch.zeros(3, int(h/self.down), int(w/self.down))
        semanMap[1,:,:] = 1

        if len(igs) > 0:
            igs = igs/self.down
            for ind in range(len(igs)):
                x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
                semanMap[1, y1:y2, x1:x2] = 0
        if len(gts)>0:
            gts = gts/self.down
            for ind in range(len(gts)):
                x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
                c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
                dx = self.gaussian(x2-x1)
                dy = self.gaussian(y2-y1)
                gau_map = torch.Tensor(np.multiply(dy, np.transpose(dx)))
                semanMap[0, y1:y2, x1:x2] = torch.max(semanMap[0, y1:y2, x1:x2], gau_map)
                semanMap[1, y1:y2, x1:x2] = 1
                semanMap[2, c_y, c_x] = 1

                if self.scale == 'h':
                    scaleMap[0, c_y-self.r:c_y+self.r+1, c_x-self.r:c_x+self.r+1] = np.log(gts[ind, 3] - gts[ind, 1])
                    scaleMap[1, c_y-self.r:c_y+self.r+1, c_x-self.r:c_x+self.r+1] = 1
                elif self.scale=='w':
                    scaleMap[0, c_y-self.r:c_y+self.r+1, c_x-self.r:c_x+self.r+1] = np.log(gts[ind, 2] - gts[ind, 0])
                    scaleMap[1, c_y-self.r:c_y+self.r+1, c_x-self.r:c_x+self.r+1] = 1
                elif self.scale=='hw':
                    scaleMap[0, c_y-self.r:c_y+self.r+1, c_x-self.r:c_x+self.r+1] = np.log(gts[ind, 3] - gts[ind, 1])
                    scaleMap[1, c_y-self.r:c_y+self.r+1, c_x-self.r:c_x+self.r+1] = np.log(gts[ind, 2] - gts[ind, 0])
                    scaleMap[2, c_y-self.r:c_y+self.r+1, c_x-self.r:c_x+self.r+1] = 1
                if self.offset:
                    offsetMap[0, c_y, c_x] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                    offsetMap[1, c_y, c_x] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                    offsetMap[2, c_y, c_x] = 1

        if self.offset:
            return img, semanMap, scaleMap, offsetMap
        else:
            return img, semanMap, scaleMap
    
    def __len__(self):
        return len(self.keys)
        
    @staticmethod
    def gaussian(kernel):
        sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
        s = 2*(sigma**2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx,(-1,1))