import random
import math
import numbers
import numpy as np

from src.database.transform.randaugment import RandAugment
from src.database.transform.normalize import Normalize
from src.database.transform.random_hflip import RandomHFlip
from src.database.transform.resize import Resize
from src.database.transform.resize_keep_aspect_ratio import ResizeKeepAspectRatio
from src.database.transform.tensorize import Tensorize
from src.database.transform.random_scale import RandScale
from src.database.transform.augmix import AugMix
from src.database.transform.random_crop import RandCrop
from src.database.transform.random_erasing import RandomErasing
from src.database.transform.random_figure import RandomFigures
from src.database.transform.random_padding import RandomPadding
from src.database.transform.random_colorjitter import RandomColorJitter
from src.database.transform.random_grayscale import RandomGrayScale
from src.database.transform.random_grid import RandomGrid
from src.database.transform.resize_fit import ResizeFit

import logging
logger = logging.getLogger("logger")

class TransformFactory:
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
        'RandomErasing',
        'RandomFigures',
        'RandomPadding',
        'RandomColorJitter',
        'RandomGrayScale',
        'RandomGrid',
        'ResizeFit',
    ]

    @classmethod
    def get_products(cls):
        return list(cls.products)

    @classmethod
    def produce(cls, cfg, trans):
        trans = trans.split(" ")
        bag_of_transforms = []    
        for tran in trans:
            if tran != "" and tran.split("-")[0] not in cls.products:
                raise KeyError("Invalid transform, got '{}', but expected to be one of {}".format(tran, cls.products))
            
            if 'RandAugment' == tran:
                bag_of_transforms.append(RandAugment(cfg.INPUT.RAND_AUG_N, cfg.INPUT.RAND_AUG_M, size=cfg.INPUT.SIZE, stride=cfg.MODEL.STRIDES[0]))

            if 'Resize' == tran:
                bag_of_transforms.append(Resize(size=cfg.INPUT.SIZE))

            if 'ResizeKeepAspectRatio' == tran:
                bag_of_transforms.append(ResizeKeepAspectRatio(size=cfg.INPUT.SIZE))

            if 'RandomHFlip' == tran:
                bag_of_transforms.append(RandomHFlip(num_keypoints=cfg.DB.NUM_KEYPOINTS))
                
            if 'Tensorize' == tran:
                bag_of_transforms.append(Tensorize())

            if 'Normalize' == tran:
                bag_of_transforms.append(Normalize(mean=cfg.INPUT.MEAN, std=cfg.INPUT.STD))
            
            if 'RandScale' in tran:
                min_s, max_s = list(map(float, tran.split('-')[1:]))
                bag_of_transforms.append(RandScale(size=cfg.INPUT.SIZE, scale(min_s, max_s)))

            if 'AugMix' in tran:
                extra = tran.split('-')[1:]
                if len(extra) > 0: 
                    assert len(extra) == 3
                    op_name, p, value = extra
                    bag_of_transforms.append(AugMix(
                        size=cfg.INPUT.SIZE,
                        stride=cfg.MODEL.STRIDES[0],
                        p=float(p),
                        op_name=op_name,
                        value=float(value)
                    ))
                else:
                    bag_of_transforms.append(AugMix(size=cfg.INPUT.SIZE, stride=cfg.MODEL.STRIDES[0]))

            if 'RandCrop' == tran:
                bag_of_transforms.append(RandCrop(size=cfg.INPUT.SIZE, pad=cfg.INPUT.PAD))

            if 'RandomErasing' == tran:
                p = float(tran.split('-')[-1])
                bag_of_transforms.append(RandomErasing(p=p))

            if 'RandomFigures' in tran:
                p = float(tran.split('-')[-1])
                bag_of_transforms.append(RandomFigures(p=p))

            if 'RandomPadding' in tran:
                p = float(tran.split('-')[-1])
                bag_of_transforms.append(RandomPadding(p=p))

            if 'RandomColorJitter' in tran:
                p, b, c, s, h = list(map(float, tran.split('-')[1:]))
                bag_of_transforms.append(RandomColorJitter(p=p, brightness=b, contrast=c, saturation=s, hue=h))

            if 'RandomGrayScale' in tran:
                p = float(tran.split('-')[-1])
                bag_of_transforms.append(RandomGrayScale(p=p))

            if 'RandomGrid' in tran:
                p = float(tran.split('-')[-1])
                bag_of_transforms.append(RandomGrid(p=p))
            
            if 'ResizeFit' == tran:
                bag_of_transforms.append(ResizeFit(divisor=cfg.MODEL.MAX_STRIDE))
            
        print(bag_of_transforms)
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
    
    def __call__(self, img, bboxes=None, ptss=None, cls_ids=None):
        '''
        Apply transformation on data like call a function, bboxes and keypoints are changed in place

        Args:
            img (PIL Image): data on which applied transformations
            bboxes (list): optional, if bboxes is given, apply transformation related to change of position
            ptss (list): list of a list with class of keypoints (int) and keypoints (Nx3 numpy array),
                        [pts1, pts2, ...], pts[:,:2] is position, pts[:,2] indicates the visibility of each pt 
                        in pts. 2 is visible, 1 is occlusion and 0 is not labeled.
            cls_ids (list): list of category of object.
        Return:
            img (PIL Image): transformed data
            ss (list): states of randomness
        '''
        ss = {}
        for t in self.t_list:
            img, s = t.apply_image(img)
            ss[t.op_name] = s

        if bboxes is not None:
            for t in self.t_list:
                for i in range(len(bboxes)):
                    bboxes[i] = t.apply_bbox(bboxes[i], ss[t.op_name])

        if ptss is not None:
            assert cls_ids is not None
            for t in self.t_list:
                for i in range(len(ptss)):
                    ptss[i] = t.apply_pts(cls_ids[i], ptss[i], ss[t.op_name])
        if bboxes is None:
            return img
        return img, ss
