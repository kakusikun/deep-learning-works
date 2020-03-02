import random

import numpy as np
import math
from src.database.transform import *
from database.transform import augmentations as aug

class RandAugment(BaseTransform):
    '''
    RandAugment is from the paper https://arxiv.org/abs/1909.13719
    Reference:
        https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py
        https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
        https://github.com/rwightman/pytorch-image-models/blob/e39aae56b4e6e3cf86c364ac71389e37266aa674/timm/data/auto_augment.py
    
    Args:
        n (int): randomly select n in 16 operations to transform data
        m (int): integer in [0, 30], apply operation on data with magnitude m
    
    '''
    def __init__(self, n, m):
        self.n = n
        self.m = m 
    
    def apply_image(self, img):        
        ops = random.choices(aug.RANDAUG_OPS_NAME, k=self.n)
        s = {}
        for op_name in ops:
            level = aug.AUG_LEVELS[op_name](self.m, size = img.size)
            img = aug.AUG_OPS[op_name](img, level)
            s[op_name] = {'level':level, 'shape':img.size}

        return img, s
    
    def apply_bbox(self, bbox, s):
        for op_name in s:
            A = aug.AUG_AS[op_name](**s[op_name])
            bbox[:2] = aug.apply_A(bbox[:2], A)
            bbox[2:] = aug.apply_A(bbox[2:], A)
        return bbox
    
    def apply_pts(self, cid, pts, s):
        for op_name in s:
            A = aug.AUG_AS[op_name](**s[op_name])
            for i in range(pts.shape[0]):
                pts[i] = aug.apply_A(pts[i], A)
        return pts

