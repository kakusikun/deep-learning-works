import torch
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image
from database.transform.base_transform import BaseTransform

class Cutout(BaseTransform):
    '''
    To normalize the data

    Args:
        mean (list): the value substracted from data
        std (list): the value divided from data

    Attributes:
        mean (list): arg, mean
        std (list): arg, std
    '''
    
    def __init__(self, length):
        self.length = length
    
    def apply_image(self, img):
        '''
        Normalize image
        Args:
            img (torch.Tensor): data to be normalized
        Return:
            img (torch.Tensor): normalized data
        '''
        assert isinstance(img, torch.Tensor)

        # If the dtype of img is float before tensorizing, 
        # the img is transformed to tensor with scale [0, 255].
        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w))
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = mask.expand_as(img)
        img *= mask

        s = {'pos': (x1, y1, x2, y2)}

        return img, s