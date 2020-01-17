import torch
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image

class Normalize():
    '''
    To normalize the data

    Args:
        mean (list): the value substracted from data, 
        std (list): the value divided from data

    Attributes:
        mean (list): arg, mean
        std (list): arg, std
    '''

    def __init__(self, mean, std):
        assert max(mean) <= 1.0
        assert max(std) <= 1.0
        self.mean = mean
        self.std = std
    
    def apply_image(self, img):
        '''
        Normalize image
        Args:
            img (PIL image): image to be normalized
        Return:
            img (PIL image): normalized image
        '''
        assert isinstance(img, torch.Tensor)
        assert img.max() <= 1.0

        img = TF.normalize(img, self.mean, self.std)
        return img
    
    def apply_bbox(self, bbox, s):
        return bbox

    def apply_pts(self, cid, pts, s):
        return pts