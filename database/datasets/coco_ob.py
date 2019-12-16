from database.datasets import *
import numpy as np
import cv2
import math
from tools.image import get_affine_transform, affine_transform, draw_umich_gaussian, gaussian_radius, color_aug

class build_coco_dataset(data.Dataset):
    # DeepFastion2 KeyPoints
    def __init__(self, data_coco, data, split):
        self.coco = data_coco
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.num_classes = len(cats)
        self.max_objs = 128
        self.default_res = (512, 512)    
        self.images = data
        self.split = split
        self.mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)
        self.cat_ids = {v: i for i, v in enumerate(self.coco.getCatIds())}
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([[-0.58752847, -0.69563484,  0.41340352],
                                  [ -0.5832747,  0.00994535, -0.81221408],
                                  [-0.56089297,  0.71832671,  0.41158938]], dtype=np.float32)

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i
    
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        img_id, img_path = self.images[index]        
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)   
        
        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]    
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        input_h, input_w = self.default_res
        # image position augmentation
        # scale
        flipped = False
        if self.split == 'train':
            s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = self._get_border(128, img.shape[1])
            h_border = self._get_border(128, img.shape[0])
            # location
            c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
            c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            # hflip
            
            if np.random.random() < 0.5:
                flipped = True
                img = img[:, ::-1, :]
                c[0] =  width - c[0] - 1
        # processing
        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        # rescale image pixel value to [0,1]
        inp = (inp.astype(np.float32) / 255.)
        # image color augmentation       
        if self.split == 'train' and np.random.random() < 0.5:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        # image normalize
        inp = (inp - self.mean) / self.std
        # HWC => CHW
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // 8
        output_w = input_w // 8

        num_classes = self.num_classes
        # transform that applying to gt
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        # center, object heatmap
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        # object size
        wh             = np.zeros((self.max_objs, 2), dtype=np.float32)
        # object offset
        reg            = np.zeros((self.max_objs             , 2             ), dtype=np.float32)       
        ind            = np.zeros((self.max_objs             ), dtype=np.int64)
        reg_mask       = np.zeros((self.max_objs             ), dtype=np.uint8)                       
            
        draw_gaussian = draw_umich_gaussian

        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            
            cls_id = int(self.cat_ids[ann['category_id']])
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]            
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1  
        ret = {'inp': inp,
               'hm': hm, 'wh':wh, 'reg':reg,
               'reg_mask': reg_mask, 'ind': ind,
               'img_id': img_id, 'c': c, 's': s}
        return ret