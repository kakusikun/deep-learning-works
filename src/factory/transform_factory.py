import random
import math
import numbers
import numpy as np
from src.base_factory import BaseFactory

from src.database.transform import (
    RandAugment, 
    Normalize, 
    RandomHFlip, 
    Tensorize, 
    ResizeKeepAspectRatio, 
    Resize, 
    RandScale, 
    AugMix, 
    RandCrop,
    Cutout
)

import logging
logger = logging.getLogger("logger")

class TransformFactory(BaseFactory):
    products = [
        'RandAugment',
        'Resize',
        'RandomHFlip',
        'ResizeKeepAspectRatio',
        'Tensorize',
        'Normalize',
        'RandScale',
        'AugMix',
        'RandCrop',
        'Cutout'
    ]
        
    @classmethod
    def produce(cls, cfg, trans):
        trans = trans.split(" ")
        bag_of_transforms = []    

        for tran in trans:
            if tran not in cls.products:
                raise KeyError("Invalid transform, got '{}', but expected to be one of {}".format(tran, cls.products))
            
            if tran == 'RandAugment':
                bag_of_transforms.append(RandAugment(cfg.INPUT.RAND_AUG_N, cfg.INPUT.RAND_AUG_M))

            if tran == 'Resize':
                bag_of_transforms.append(Resize(size=cfg.INPUT.RESIZE, stride=cfg.MODEL.STRIDE))

            if tran == 'ResizeKeepAspectRatio':
                bag_of_transforms.append(ResizeKeepAspectRatio(size=cfg.INPUT.RESIZE, stride=cfg.MODEL.STRIDE))

            if tran == 'RandomHFlip':
                bag_of_transforms.append(RandomHFlip(stride=cfg.MODEL.STRIDE, num_keypoints=cfg.DB.NUM_KEYPOINTS))
                
            if tran == 'Tensorize':
                bag_of_transforms.append(Tensorize())

            if tran == 'Normalize':
                bag_of_transforms.append(Normalize(mean=cfg.INPUT.MEAN, std=cfg.INPUT.STD))
            
            if tran == 'RandScale':
                bag_of_transforms.append(RandScale(size=cfg.INPUT.RESIZE, stride=cfg.MODEL.STRIDE))

            if tran == 'AugMix':
                bag_of_transforms.append(AugMix())

            if tran == 'RandCrop':
                bag_of_transforms.append(RandCrop(size=cfg.INPUT.RESIZE, pad=cfg.INPUT.PAD))

            if tran == 'Cutout':
                bag_of_transforms.append(Cutout(length=16))

        return Transform(bag_of_transforms)

class Transform():
    '''
    To compose the transformations that apply on data. 
    Works like torchvision transform Compose.

    Args:
        t_list (list): an list of transformations that apply on data in order
    '''
    def __init__(self, t_list):
        self.t_list = t_list
    
    def __call__(self, img, bboxes=None, ptss=None):
        '''
        Apply transformation on data like call a function, bboxes and keypoints are changed in place

        Args:
            img (PIL Image): data on which applied transformations
            bboxes (list): optional, if bboxes is given, apply transformation related to change of position
            ptss (list): optional, works like bboxes, but each cell in list is a list with class of keypoints and keypoints.
                              [[c1, pts1], [c2, pts2], ...]
        
        Return:
            img (PIL Image): transformed data
            ss (list): states of randomness
        '''
        ss = {}
        for t in self.t_list:
            img, s = t.apply_image(img)
            ss[t] = s

        if bboxes is not None:
            for t in self.t_list:
                for i in range(len(bboxes)):
                    bboxes[i] = t.apply_bbox(bboxes[i], ss[t])

        if ptss is not None:
            for t in self.t_list:
                for i in range(len(ptss)):
                    cls_id, pts = ptss[i]
                    ptss[i][1] = t.apply_pts(cls_id, pts, ss[t])
        if bboxes is None:
            return img
        return img, ss
