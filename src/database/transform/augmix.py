import random

import numpy as np
import math
from src.database.transform import *
import src.database.transform.augmentations as aug

class AugMix(BaseTransform):
    '''
    Perform AugMix augmentations and compute mixture (https://arxiv.org/abs/1912.02781).
    Reference : 
        AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty
        https://github.com/google-research/augmix
                
    Args:
        width (int): optional, number of augmentation chains applied on image, default is 3
        depth (int): optional, length of augmentations to form a augmentation chain, 
                     default is -1 which uses random length in [1, 3]
        mag (int): optional, the severities of augmentation, default is 3 which uses random
                   severities in [0.1, 3]
    '''

    def __init__(self, size, stride, width=3, depth=-1, mag=3, p=0.5, op_name=None, value=0):
        self.size = size
        self.stride = stride
        self.width = width
        self.depth = depth+1 if depth > 0 else 4
        self.mag = mag
        self.p = p
        self.op_name = op_name
        self.value = value
        

    def apply_image(self, img):   
        '''
        Args:
            image (PIL.Image): input image
        Returns:
            mixed (numpy.ndarray): Augmented and mixed image.
        '''
        s = {} 
        if self.op_name is None:
            ws = np.random.dirichlet([1] * self.width).astype(np.float32)
            m = np.random.beta(1, 1)
            mix = np.zeros_like(np.array(img), dtype=np.float32)
            for i in range(self.width):
                depth = np.random.randint(1, self.depth)
                for _ in range(depth):
                    img_aug = img.copy()                
                    op_name = np.random.choice(aug.AUGMIX_OPS_NAME)
                    mag = np.random.uniform(low=0.1, high=self.mag)
                    level = aug.AUG_LEVELS[op_name](mag, size=img.size)
                    img_aug = aug.AUG_OPS[op_name](img_aug, level)
                    s[op_name] = {'level':level, 'shape':img.size}
                # Preprocessing commutes since all coefficients are convex
                mix += ws[i] * np.array(img_aug).astype(np.float32)
            mixed = (1 - m) * np.array(img).astype(np.float32) + m * mix
            return mixed, s
        else:
            if random.uniform(0, 1) > self.p:
                return img, s
            mag = np.random.uniform(low=0, high=10)
            if self.value > 0:
                level = aug.AUG_LEVELS[self.op_name](mag, size=img.size, value=self.value)
            else:
                level = aug.AUG_LEVELS[self.op_name](mag, size=img.size)
            img = aug.AUG_OPS[self.op_name](img, level)
            s[self.op_name] = {'level':level, 'shape':img.size}
            return img, s

    def apply_bbox(self, bbox, s):
        for op_name in s:
            A = aug.AUG_AS[op_name](**s[op_name])
            tlx, tly, brx, bry = bbox
            trx, tr_y, blx, bly = brx, tly, tlx, bry
            tlx, tly = aug.apply_A([tlx, tly], A)
            brx, bry = aug.apply_A([brx, bry], A)
            trx, tr_y = aug.apply_A([trx, tr_y], A)
            blx, bly = aug.apply_A([blx, bly], A)
            x1 = np.array([tlx, trx, blx, brx]).min()
            y1 = np.array([tly, tr_y, bly, bry]).min()
            x2 = np.array([tlx, trx, blx, brx]).max()
            y2 = np.array([tly, tr_y, bly, bry]).max()
            bbox = np.array([x1, y1, x2, y2])
        out_w, out_h = np.array(self.size)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, out_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, out_h - 1) 
        return bbox
    
    def apply_pts(self, cid, pts, s):
        out_h, out_w = (np.array(self.size) // self.stride).astype(int)
        for op_name in s:
            A = aug.AUG_AS[op_name](**s[op_name])
            for i in range(pts.shape[0]):
                pts[i,:2] = aug.apply_A(pts[i,:2], A)
                if ((pts[i, :2] < 0).sum() + (pts[i, :2] > (out_w, out_h)).sum()) > 0:
                    pts[i, 2] = 0.0
        return pts

