import random
import math
import numbers
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F

from database.transform.reid_color_aug import REID_Color_aug
from database.transform.random_erasing import RandomErasing
import logging
logger = logging.getLogger("logger")

transforms = [
    'reid_color_aug',
    'resize',
    'hflip',
    'random_crop',
    'to_tensor',
    'normalize',
    'random_erase',
    'center_crop',
]

def get_transform(cfg, trans):
    trans = trans.split(" ")
    bag_of_transforms = []    

    for tran in trans:
        if tran == 'reid_color_aug':
            bag_of_transforms.append(REID_Color_aug())

        # elif tran == 'resize':
        #     bag_of_transforms.append(T.Resize(size=cfg.INPUT.RESIZE))

        # elif tran == 'hflip':
        #     bag_of_transforms.append(T.RandomHorizontalFlip(p=cfg.INPUT.PROB))
            
        elif tran == 'random_crop':
            bag_of_transforms.append(T.RandomCrop(size=cfg.INPUT.CROP_SIZE, padding=cfg.INPUT.PAD))   

        elif tran == 'to_tensor':
            bag_of_transforms.append(T.ToTensor())

        elif tran == 'normalize':
            bag_of_transforms.append(T.Normalize(mean=cfg.INPUT.MEAN, std=cfg.INPUT.STD))

        elif tran == 'random_erase':
            bag_of_transforms.append(RandomErasing())     
                   
        elif tran == 'center_crop':
            bag_of_transforms.append(T.CenterCrop(size=cfg.INPUT.CROP_SIZE))
        else:            
            raise KeyError("Invalid transform, got '{}', but expected to be one of {}".format(tran, transforms))
                
    transform = T.Compose(bag_of_transforms)

    return transform