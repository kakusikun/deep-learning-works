from database.datasets import *
import numpy as np
from PIL import Image

class build_coco_dataset(data.Dataset):
    def __init__(self, data, trans=None, build_func=None):
        self.coco = data['handle']
        self.num_classes = data['num_classes']
        self.num_keypoints = data['num_keypoints']
        self.stride = data['stride']
        self.max_objs = 32
        self.indice = data['indice']
        self.cat_ids = {v: i for i, v in enumerate(self.coco.getCatIds())}
        self.trans = trans
        self.build_func = build_func

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
        return bbox
   
    def __len__(self):
        return len(self.indice)
        
    def __getitem__(self, index):
        img_id, img_path = self.indice[index]        
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)  

        use_kp = False
        if len(anns) > 0 and 'keypoints' in anns[0] and len(anns[0]['keypoints']) > 0:
            use_kp = True
        
        ret = {}
        ss = {}
        img = Image.open(img_path)
        bboxes = []
        ptss = []
        valid_ptss = []

        for k in range(num_objs):
            ann = anns[k]
            bboxes.append(self._coco_box_to_bbox(ann['bbox']))            
            cls_id = int(self.cat_ids[ann['category_id']])
            if use_kp:
                pts = np.array(ann['keypoints'], np.float32).reshape(self.num_keypoints, 3)
            else:
                pts = np.zeros((self.num_keypoints, 3))
            ptss.append([cls_id, pts[:,:2]])
            valid_ptss.append(pts[:,2])

        # rescale => hflip => tensorize => normalize
        if self.trans is not None:
            if use_kp:
                img, ss = self.trans(img, bboxes=bboxes, ptss=ptss)
            else:
                img, ss = self.trans(img, bboxes=bboxes)

        if isinstance(img, Image.Image):
            in_w, in_h = img.size    
        else:
            in_w, in_h = img.shape[2], img.shape[1]
        outsize = (in_w // self.stride, in_h // self.stride)

        if self.build_func is not None:
            ret = self.build_func(
                bboxes=bboxes, 
                ptss=ptss, 
                valid_ptss=valid_ptss, 
                max_objs=self.max_objs,
                num_classes=self.num_classes, 
                num_keypoints=self.num_keypoints,
                outsize=outsize
            )
        else:            
            ret['bboxes'] = bboxes
            if use_kp:
                ret['ptss'] = ptss

        ret['inp'] = img

        if 'RandScale' in ss:
            ret['c'] = ss['RandScale']['c']
            ret['s'] = ss['RandScale']['s']
        else:
            if isinstance(img, Image.Image):
                w, h = img.size()
            else:
                h, w = img.shape[1:]
            ret['c'] = np.array([w / 2., h / 2.], dtype=np.float32)
            ret['s'] = max(h, w) * 1.0

        ret['img_id'] = img_id

        return ret