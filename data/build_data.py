
import torch
import os
import os.path as osp
import sys
import cv2
import numpy as np
import torch.utils.data as data
import math
from PIL import Image
import cv2
# from data import caffe_pb2
from data.build_transform import RandomErasing, _RandomCrop, _RandomHorizontalFlip
import torchvision.transforms as T
import torchvision.transforms.functional as F
from tools.image import get_affine_transform, affine_transform, draw_umich_gaussian, gaussian_radius


class build_image_dataset(data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, index):
        img_path, label = self.dataset[index]

        img = Image.open(img_path)        
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        return len(self.dataset)


class build_reid_dataset(data.Dataset):
    def __init__(self, dataset, transform=None, return_indice=False):
        self.dataset = dataset
        self.transform = transform
        self.return_indice = return_indice
           
    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]

        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)
            if isinstance(img, tuple):
                img = img[0]
        if self.return_indice:
            return img, pid, camid, index
            
        return img, pid, camid

    def __len__(self):
        return len(self.dataset)

class build_update_reid_dataset(data.Dataset):
    def __init__(self, new_labels, dataset, transform=None):
        self.dataset = dataset
        self.new_labels = new_labels
        self.transform = transform
        self.update_dataset()
           
    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]

        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, pid, camid
    
    def __len__(self):
        return len(self.dataset)

    def update_dataset(self):
        new_dataset = []
        for i, (img_path, _, _) in enumerate(self.dataset):
            label = self.new_labels[0][i]
            if label == -1:
                continue
            new_dataset.append((img_path, label, 0))
        self.dataset = new_dataset

class build_reid_atmap_dataset(data.Dataset):
    def __init__(self, dataset, cfg):
        self.dataset = dataset
        self.at_maps = cfg.DATASET.ATTENTION_MAPS
        self.at_maps_keys = {}
        with open(cfg.DATASET.ATTENTION_MAPS_LIST, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                self.at_maps_keys[line] = i
        
        if cfg.TRANSFORM.RESIZE:
            self.resize = T.Resize(size=cfg.INPUT.IMAGE_SIZE)
            self.at_map_resize = T.Resize(size=cfg.INPUT.IMAGE_SIZE, interpolation=Image.NEAREST)
        else:
            self.resize = None
            self.at_map_resize = None

        if cfg.TRANSFORM.HFLIP:
            self.random_hflip = _RandomHorizontalFlip(p=cfg.INPUT.PROB)
        else:
            self.random_hflip = None

        if cfg.TRANSFORM.RANDOMCROP:
            self.random_crop = _RandomCrop(size=cfg.INPUT.IMAGE_CROP_SIZE, padding=cfg.INPUT.IMAGE_PAD)
        else:
            self.random_crop = None
        
        if cfg.TRANSFORM.NORMALIZE:
            self.normalize = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        else:
            self.normalize = None

        if cfg.TRANSFORM.RANDOMERASING:
            self.random_erase = RandomErasing()
        else:
            self.random_erase = None
           
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
        if self.resize is not None:
            img = self.resize(img)
        if self.random_hflip is not None:
            img, is_flip = self.random_hflip(img)
        if self.random_crop is not None:
            img, i, j, h, w = self.random_crop(img)
        img = F.to_tensor(img)
        if self.normalize is not None:
            img = self.normalize(img)
        if self.random_erase is not None:
            img, x1, y1, rh, rw = self.random_erase(img)
        
        if at_map_label > 0:
            at_map = self.at_map_resize(at_map)
            if self.random_hflip is not None:
                at_map = F.hflip(at_map)
            if self.random_crop is not None:
                at_map = self.random_crop.by_param(at_map, i, j, h, w)
            at_map = torch.Tensor(np.array(at_map)).transpose(0,2).transpose(1,2)
            if self.random_erase is not None:
                at_map = self.random_erase.by_param(at_map, x1, y1, rh, rw)
            at_map = at_map.transpose(0,2).transpose(0,1).numpy()
            at_map = cv2.resize(at_map, (8, 16), interpolation=cv2.INTER_NEAREST).mean(axis=2)
            at_map = torch.Tensor(at_map).view(-1)    
        
        return img, pid, camid, at_map, at_map_label
    
    def __len__(self):
        return len(self.dataset)

class build_par_dataset(data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
           
    def __getitem__(self, index):
        img_path, attrs = self.dataset[index]

        img = Image.open(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        attrs = torch.Tensor(attrs)

        return img, attrs
    
    def __len__(self):
        return len(self.dataset)

class build_COCO_Person_dataset(data.Dataset):
    # DeepFastion2 KeyPoints
    def __init__(self, data_coco, data, split):
        self.coco = data_coco
        self.num_classes = 1
        self.max_objs = 32
        self.default_res = (512, 512)    
        self.images = data
        self.split = split
        self.mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)
        
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _prerocessing(self, path, is_train=True):
        if is_train:
            img = cv2.imread(path)
            height, width = img.shape[:2]         
            long_side = np.max(img.shape[:2])
            canvas = np.zeros([long_side, long_side, 3])
            h_offset, w_offset = int((long_side-height)/2), int((long_side-width)/2)
            canvas[h_offset:(height+h_offset), w_offset:(width+w_offset), :] = img
            img = canvas.astype(np.uint8)   
            scale = self.default_res[0] / long_side        
            img = cv2.resize(img, self.default_res) 
        else:
            img = cv2.imread(path)
            h, w = img.shape[:2]         
            h_offset, w_offset = 16 - (h % 16), 16 - (w % 16)
            canvas = np.zeros([h + h_offset, w + w_offset, 3])
            canvas[:h, :w, :] = img
            img = canvas.astype(np.uint8)
            scale = 1.0
        return img, w_offset, h_offset, scale
    
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        img_id, img_path = self.images[index]        
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)        

        if self.split == 'train':
            # keep aspect ratio to resize to default resolution
            img, w_offset, h_offset, scale = self._prerocessing(img_path)
            c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
            s = max(img.shape[0], img.shape[1]) * 1.0
            sf = 0.4
            cf = 0.1
            c[0] = c[0] + s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
            c[1] = c[1] + s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            trans_input = get_affine_transform(c, s, 0, self.default_res)
            inp = cv2.warpAffine(img, trans_input, self.default_res, flags=cv2.INTER_LINEAR)

        else:
            inp, w_offset, h_offset, scale = self._prerocessing(img_path, is_train=False) 
            c = np.array([(inp.shape[1]-w_offset) / 2., (inp.shape[0]-h_offset) / 2.], dtype=np.float32)
            s = max((inp.shape[0]-h_offset), (inp.shape[1]-w_offset)) * 1.0        

        if self.split == 'train':
            output_res = self.default_res[0] // 8
            num_classes = self.num_classes
            trans_output = get_affine_transform(c, s, 0, [output_res, output_res])
        else:
            output_res_w, output_res_h = inp.shape[1] // 8, inp.shape[0] // 8
            num_classes = self.num_classes
            trans_output = get_affine_transform(c, s, 0, [output_res_w, output_res_h])

        # [0,1]
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        # HWC => CHW
        inp = inp.transpose(2, 0, 1)

        # center, object heatmap
        if self.split == 'train':
            hm = np.zeros((num_classes, output_res, output_res), dtype=np.float32)
        else:
            hm = np.zeros((num_classes, output_res_h, output_res_w), dtype=np.float32)
        # object size
        wh             = np.zeros((self.max_objs, 2), dtype=np.float32)
        # object offset
        reg            = np.zeros((self.max_objs             , 2             ), dtype=np.float32)       
        ind            = np.zeros((self.max_objs             ), dtype=np.int64)
        reg_mask       = np.zeros((self.max_objs             ), dtype=np.uint8)                       
            
        draw_gaussian = draw_umich_gaussian

        for k in range(num_objs):
            ann = anns[k]
            if ann['category_id'] != 1:
                continue
            bbox = self._coco_box_to_bbox(ann['bbox'])
            if self.split == 'train':
                bbox[[0, 2]] += w_offset
                bbox[[1, 3]] += h_offset
                bbox *= scale

            cls_id = 0
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            if self.split == 'train':
                bbox = np.clip(bbox, 0, output_res - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]            
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                if self.split == 'train':
                    ind[k] = ct_int[1] * output_res + ct_int[0]
                else:
                    ind[k] = ct_int[1] * output_res_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1  

        ret = {'inp': inp, 
               'hm': hm, 'wh':wh, 'reg':reg,
               'reg_mask': reg_mask, 'ind': ind}

        if self.split != 'train':            
            ret.update({'img_id': img_id, 'c': c, 's': s})
        return ret

class build_DFKP_dataset(data.Dataset):
    # DeepFastion2 KeyPoints
    def __init__(self, data_handle, data, src, split):
        self.coco = data_handle
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.num_classes = len(cats)
        self.num_joints = 294
        self.max_objs = 32
        self.default_res = (512, 512)    
        self.images = data
        self.src = src
        self.split = split
        self.mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)
        
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _prerocessing(self, path):
        img = cv2.imread(path)
        height, width = img.shape[:2]         
        long_side = np.max(img.shape[:2])
        canvas = np.zeros([long_side, long_side, 3])
        h_offset, w_offset = int((long_side-height)/2), int((long_side-width)/2)
        canvas[h_offset:(height+h_offset), w_offset:(width+w_offset), :] = img
        img = canvas.astype(np.uint8)   
        scale = self.default_res[0] / long_side        
        img = cv2.resize(img, self.default_res) 
        
        return img, w_offset, h_offset, scale
    
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        img_id = self.images[index]
        fname = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = osp.join(self.src, fname)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)
        

        if self.split == 'train':
            # keep aspect ratio to resize to default resolution
            img, w_offset, h_offset, scale = self._prerocessing(img_path)
            height, width = img.shape[0], img.shape[1]
            c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
            s = max(img.shape[0], img.shape[1]) * 1.0
            sf = 0.4
            cf = 0.1
            c[0] = c[0] + s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
            c[1] = c[1] + s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)     

            trans_input = get_affine_transform(c, s, 0, self.default_res)
            inp = cv2.warpAffine(img, trans_input, self.default_res, flags=cv2.INTER_LINEAR)
            # [-1,1]
            inp = (inp.astype(np.float32) / 255.)
            inp = (inp - self.mean) / self.std
            # HWC => CHW
            inp = inp.transpose(2, 0, 1)

            output_res = self.default_res[0] // 8
            num_joints = self.num_joints
            num_classes = self.num_classes
            trans_output = get_affine_transform(c, s, 0, [output_res, output_res])
            
            # center, object heatmap
            hm             = np.zeros((num_classes, output_res, output_res), dtype=np.float32)
            # human pose heatmap
            hm_hp          = np.zeros((num_joints , output_res, output_res), dtype=np.float32)

            # object size
            wh             = np.zeros((self.max_objs, 2), dtype=np.float32)
            # object offset
            reg            = np.zeros((self.max_objs             , 2             ), dtype=np.float32)       
            # humna pose offset
            hp_offset      = np.zeros((self.max_objs * num_joints, 2             ), dtype=np.float32)
            # human pose location relative to center
            kps            = np.zeros((self.max_objs             , num_joints * 2), dtype=np.float32)
            kps_mask       = np.zeros((self.max_objs             , num_joints * 2), dtype=np.uint8)
            
            ind            = np.zeros((self.max_objs             ), dtype=np.int64)
            reg_mask       = np.zeros((self.max_objs             ), dtype=np.uint8) 
            hp_ind         = np.zeros((self.max_objs * num_joints), dtype=np.int64)
            hp_mask        = np.zeros((self.max_objs * num_joints), dtype=np.int64)            
            
            
            draw_gaussian = draw_umich_gaussian

            for k in range(num_objs):
                ann = anns[k]
                bbox = self._coco_box_to_bbox(ann['bbox'])
                bbox[[0, 2]] += w_offset
                bbox[[1, 3]] += h_offset
                bbox *= scale

                cls_id = int(ann['category_id']) - 1
                pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3)
                pts[:,0] += w_offset
                pts[:,1] += h_offset
                pts[:,:2] *= scale

                bbox[:2] = affine_transform(bbox[:2], trans_output)
                bbox[2:] = affine_transform(bbox[2:], trans_output)
                bbox = np.clip(bbox, 0, output_res - 1)
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                
                if h > 0 and w > 0:
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    wh[k] = 1. * w, 1. * h
                    ind[k] = ct_int[1] * output_res + ct_int[0]
                    reg[k] = ct - ct_int
                    reg_mask[k] = 1
                    num_kpts = pts[:, 2].sum()
                    if num_kpts == 0:
                        hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
                        reg_mask[k] = 0

                    hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    hp_radius = max(0, int(hp_radius)) 
                    for j in range(num_joints):
                        if pts[j, 2] > 0:
                            pts[j, :2] = affine_transform(pts[j, :2], trans_output)
                            if pts[j, 0] >= 0 and pts[j, 0] < output_res and pts[j, 1] >= 0 and pts[j, 1] < output_res:
                                kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                                kps_mask[k, j * 2: j * 2 + 2] = 1
                                pt_int = pts[j, :2].astype(np.int32)
                                hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
                                hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]
                                hp_mask[k * num_joints + j] = 1
                                draw_gaussian(hm_hp[j], pt_int, hp_radius)
                    draw_gaussian(hm[cls_id], ct_int, radius)
                    
            
            ret = {   'input': inp, 
                         'hm': hm,           'wh': wh,         'reg': reg,
                   'reg_mask': reg_mask,    'ind': ind,
                      'hm_hp': hm_hp,       'hps': kps,  'hp_offset': hp_offset,
                   'hps_mask': kps_mask, 'hp_ind': hp_ind, 'hp_mask': hp_mask}
        else:
            img = cv2.imread(img_path)
            h, w = img.shape[:2]         
            h_offset, w_offset = h % 16, w % 16
            canvas = np.zeros([h + h_offset, w + w_offset, 3])
            canvas[:h, :w, :] = img
            img = canvas.astype(np.uint8)  
            # [-1,1]
            inp = (img.astype(np.float32) / 255.)
            inp = (inp - self.mean) / self.std
            # HWC => CHW
            inp = inp.transpose(2, 0, 1)
            ret = {'input': inp}

        return ret


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
