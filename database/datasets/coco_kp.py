from database.datasets import *
import numpy as np
from PIL import Image
from tools.centernet_utils import CenterNet_keypoints_target

class build_cocokp_dataset(data.Dataset):
    def __init__(self, data, trans=None):
        self.coco = data['handle']
        self.num_classes = data['num_classes']
        self.num_joints = data['num_keypoints']
        self.stride = data['stride']
        self.max_objs = 32
        self.indice = data['indice']
        self.cat_ids = {v: i for i, v in enumerate(self.coco.getCatIds())}
        self.trans = trans

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
        
        img = Image.open(img_path)
        bboxes = []
        ptss = []
        valid_ptss = []

        for k in range(num_objs):
            ann = anns[k]
            bboxes.append(self._coco_box_to_bbox(ann['bbox']))            
            cls_id = int(self.cat_ids[ann['category_id']])
            pts = np.array(ann['keypoints'], np.float32).reshape(self.num_joints, 3)
            ptss.append([cls_id, pts[:,:2]])
            valid_ptss.append(pts[:,2])

        # rescale => hflip => tensorize => normalize
        if self.trans is not None:
            img, ss = self.trans(img, bboxes, ptss)

        if isinstance(img, Image.Image):
            in_w, in_h = img.size    
        else:
            in_w, in_h = img.shape[2], img.shape[1]
        outsize = (in_w // self.stride, in_h // self.stride)

        ret = CenterNet_keypoints_target(bboxes, ptss, valid_ptss, self.max_objs, self.num_classes, self.num_joints, outsize)

        ret['inp'] = img

        if 'RandScale' in ss:
            ret['c'] = ss['RandScale']['c']
            ret['s'] = ss['RandScale']['s']
        else:
            np_img = np.array(img)
            h, w = np_img.shape[:2] 
            ret['c'] = np.array([w / 2., h / 2.], dtype=np.float32)
            ret['s'] = max(h, w) * 1.0

        ret['img_id'] = img_id

        return ret