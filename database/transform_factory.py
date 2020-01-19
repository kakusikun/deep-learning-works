import random
import math
import numbers
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F

from database.transform import RandAugment, Normalize, RandomHFlip, Tensorize, ResizeKeepAspectRatio, Resize 
import logging
logger = logging.getLogger("logger")

TRANFORMS = [
    'randaug',
    'resize',
    'hflip',
    'resize_keep_ratio',
    'tensorize',
    'normalize',
]

def get_transform(cfg, trans):
    trans = trans.split(" ")
    bag_of_transforms = []    

    for tran in trans:
        if tran not in TRANFORMS:
            raise KeyError("Invalid transform, got '{}', but expected to be one of {}".format(tran, TRANFORMS))
        
        # if tran == 'randaug':
        #     #TODO: add random augment to config
        #     bag_of_transforms.append(RandAugment())

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
                   
    return Transform(bag_of_transforms)

class Transform():
    def __init__(self, t_list):
        self.t_list = t_list
    
    def __call__(self, img, bboxes=None, total_pts=None):
        for t in self.t_list:
            img, state = t.apply_image(img)
            if bboxes is not None:
                for i in range(len(bboxes)):
                    bboxes[i] = t.apply_bbox(bboxes[i], state)
            if total_pts is not None:
                for i in range(len(total_pts)):
                    cls_id, pts = total_pts[i]
                    total_pts[i][1] = t.apply_pts(cls_id, pts, state)

