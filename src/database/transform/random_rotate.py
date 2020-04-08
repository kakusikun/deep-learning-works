import random

import numpy as np
import math
from src.database.transform import *
import src.database.transform.augmentations as aug

class RandomRotate(BaseTransform):

    def __init__(self, p, size=[0, 0]):
        self.p = p
        self.size = size
        self.op_name = 'Rotate'

    def apply_image(self, img):   
        '''
        Args:
            image (PIL.Image): input image
        Returns:
            mixed (numpy.ndarray): Augmented and mixed image.
        '''
        s = {'level':0, 'shape':img.size}
        if random.uniform(0, 1) > self.p:
            return img, s
        mag = np.random.uniform(low=0, high=10)
        level = aug.AUG_LEVELS[self.op_name](mag, size = img.size)
        img = aug.AUG_OPS[self.op_name](img, level)
        s[self.op_name] = {'level':level, 'shape':img.size}
        return img, s

    def apply_bbox(self, bbox, s):
        A = aug.AUG_AS[self.op_name](**s[self.op_name])
        bbox[:2] = aug.apply_A(bbox[:2], A)
        bbox[2:] = aug.apply_A(bbox[2:], A)
        out_w, out_h = np.array(self.size)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, out_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, out_h - 1) 
        return bbox
    
    def apply_pts(self, cid, pts, s):
        out_w, out_h = np.array(self.size)
        A = aug.AUG_AS[self.op_name](**s[self.op_name])
        for i in range(pts.shape[0]):
            pts[i,:2] = aug.apply_A(pts[i,:2], A)
            if ((pts[i, :2] < 0).sum() + (pts[i, :2] > (out_w, out_h)).sum()) > 0:
                pts[i, 2] = 0.0
        return pts

