from src.database.data_format import *
import numpy as np
from PIL import Image
from copy import deepcopy

class build_coco_dataset(Dataset):
    def __init__(self, data, transform=None, build_func=None, **kwargs):
        self.coco = data['handle'] if isinstance(data['handle'], list) else [data['handle']]
        self.pid = data['pid'] if isinstance(data['pid'], list) else [data['pid']]
        self.num_classes = data['num_classes']
        self.num_keypoints = data['num_keypoints']
        self.num_person = data['num_person']
        self.strides = data['strides']
        self.max_objs = 100
        self.indice = data['indice']
        self.cat_ids = {v: i for i, v in enumerate(self.coco[0].getCatIds())}
        self.transform = transform
        self.build_func = build_func
        self.use_kp = True if self.num_keypoints > 0 else False

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3], 1.0], dtype=np.float32)
        return bbox
    
    def __len__(self):
        return len(self.indice)
        
    def __getitem__(self, index):
        meta = self.indice[index]
        if len(meta) == 2:
            img_id, img_path = meta
            handle_idx, offset = 0, 0
        else:
            img_id, img_path, handle_idx, offset = meta

        ann_ids = self.coco[handle_idx].getAnnIds(imgIds=[img_id])
        anns = self.coco[handle_idx].loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        w, h = img.size

        ret = {}
        ss = {}        
        cls_ids = []
        bboxes = []
        ptss = []
        ids = []

        for k in range(num_objs):
            ann = anns[k]
            cls_ids.append(int(self.cat_ids[ann['category_id']]))
            bboxes.append(self._coco_box_to_bbox(ann['bbox']))
            if 'pid' in ann:
                pid = self.pid[handle_idx][ann['pid']]
            else:
                pid = -1
                
            if pid > 0:
                ids.append(pid + offset)
            else:
                ids.append(pid)

            if self.use_kp:
                pts = np.array(ann['keypoints'], np.float32).reshape(self.num_keypoints, 3)
            else:
                pts = np.zeros((self.num_keypoints, 3))
            ptss.append(pts)

        if self.transform is not None:
            while True:
                _bboxes = deepcopy(bboxes)
                _ptss = deepcopy(ptss)
                if self.use_kp:
                    _img, ss = self.transform(img, bboxes=_bboxes, ptss=_ptss, cls_ids=cls_ids)
                else:
                    _img, ss = self.transform(img, bboxes=_bboxes)
                
                num_valid = 0
                for bbox in _bboxes:
                    if (bbox == 0).sum() >= 2:
                        continue
                    num_valid += 1
                if num_valid > 0:
                    break
            img = _img
            bboxes = _bboxes
            ptss = _ptss

        if isinstance(img, Image.Image):
            in_w, in_h = img.size    
        elif isinstance(img, np.ndarray):
            in_w, in_h = img.shape[1], img.shape[0]
        elif isinstance(img, torch.Tensor):
            in_w, in_h = img.shape[2], img.shape[1]

        valid_cls_ids = []
        valid_ids = []
        valid_bboxes = []
        valid_ptss = []
        for cls_id, pid, bbox, pts in zip(cls_ids, ids, bboxes, ptss):
            if (bbox == 0).sum() >= 2:
                continue
            bbox[[0, 2]] /= in_w
            bbox[[1, 3]] /= in_h
            valid_cls_ids.append(cls_id)
            valid_ids.append(pid)
            valid_bboxes.append(bbox)
            valid_ptss.append(pts)
        
        if self.use_kp:
            for i in range(len(valid_ptss)):
                for j in range(len(valid_ptss[i])):
                    valid_ptss[i][j][0] /= in_w
                    valid_ptss[i][j][1] /= in_h
        
        out_sizes = [(in_w // stride, in_h // stride) for stride in self.strides]
        if self.build_func is not None:
            ret = self.build_func(
                cls_ids=valid_cls_ids,
                bboxes=valid_bboxes, 
                ptss=valid_ptss, 
                max_objs=self.max_objs,
                num_classes=len(self.cat_ids), 
                num_keypoints=self.num_keypoints,
                out_sizes=out_sizes,
                wh=(in_w, in_h),
                strides=self.strides,
                ids=valid_ids
            )
                      
        ret['inp'] = img
        ret['img_id'] = img_id
        ret['bboxes'] = valid_bboxes
        if self.use_kp:
            ret['ptss'] = valid_ptss

        if 'RandScale' in ss:
            ret['c'] = ss['RandScale']['c']
            ret['s'] = ss['RandScale']['s']
        else:
            ret['c'] = np.array([w / 2., h / 2.], dtype=np.float32)
            ret['s'] = max(h, w) * 1.0

        

        return ret