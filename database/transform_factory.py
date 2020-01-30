import random
import math
import numbers
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F

from database.transform import RandAugment, Normalize, RandomHFlip, Tensorize, ResizeKeepAspectRatio, Resize, RandScale
import logging
logger = logging.getLogger("logger")

TRANFORMS = [
    'randaug',
    'resize',
    'hflip',
    'resize_keep_ratio',
    'tensorize',
    'normalize',
    'rescale',
]

def get_transform(cfg, trans):
    trans = trans.split(" ")
    bag_of_transforms = []    

    for tran in trans:
        if tran not in TRANFORMS:
            raise KeyError("Invalid transform, got '{}', but expected to be one of {}".format(tran, TRANFORMS))
        
        if tran == 'randaug':
            bag_of_transforms.append(RandAugment(cfg.INPUT.RAND_AUG_N, cfg.INPUT.RAND_AUG_M))

        if tran == 'resize':
            bag_of_transforms.append(Resize(size=cfg.INPUT.RESIZE, stride=cfg.MODEL.STRIDE))

        if tran == 'resize_keep_ratio':
            bag_of_transforms.append(ResizeKeepAspectRatio(size=cfg.INPUT.RESIZE, stride=cfg.MODEL.STRIDE))

        if tran == 'hflip':
            bag_of_transforms.append(RandomHFlip(num_keypoints=cfg.DB.NUM_KEYPOINTS))
            
        if tran == 'tensorize':
            bag_of_transforms.append(Tensorize())

        if tran == 'normalize':
            bag_of_transforms.append(Normalize(mean=cfg.INPUT.MEAN, std=cfg.INPUT.STD))
        
        if tran == 'rescale':
            bag_of_transforms.append(RandScale(size=cfg.INPUT.RESIZE, stride=cfg.MODEL.STRIDE))
                   
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
    
    def __call__(self, img, bboxes=None, total_pts=None):
        '''
        Apply transformation on data like call a function, bboxes and keypoints are changed in place

        Args:
            img (PIL Image): data on which applied transformations
            bboxes (list): optional, if bboxes is given, apply transformation related to change of position
            total_pts (list): optional, works like bboxes, but each cell in list is a list with class of keypoints and keypoints.
                              [[c1, pts1], [c2, pts2], ...]
        
        Return:
            img (PIL Image): transformed data
            ss (list): states of randomness
        '''
        ss = []
        for t in self.t_list:
            # TODO: add __repr__ for all transformaion to let the random state be accessed by string
            img, s = t.apply_image(img)
            ss.append(s)

        if bboxes is not None:
            for t, s in zip(self.t_list, ss):
                for i in range(len(bboxes)):
                    bboxes[i] = t.apply_bbox(bboxes[i], s)

        if total_pts is not None:
            for t, s in zip(self.t_list, ss):
                for i in range(len(total_pts)):
                    cls_id, pts = total_pts[i]
                    total_pts[i][1] = t.apply_pts(cls_id, pts, s)
        if bboxes is None:
            return img
        return img, ss
