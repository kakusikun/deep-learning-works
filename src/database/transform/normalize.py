import torch
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image
from src.database.transform import *

class Normalize(BaseTransform):
    '''
    To normalize the data

    Args:
        mean (list): the value substracted from data
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
            img (torch.Tensor): data to be normalized
        Return:
            img (torch.Tensor): normalized data
        '''
        assert isinstance(img, torch.Tensor)

        # If the dtype of img is float before tensorizing, 
        # the img is transformed to tensor with scale [0, 255].
        if not img.max() <= 1.0:
            if not isinstance(img, torch.ByteTensor):
                img.div_(255.0)
            else:
                raise ValueError
        
        img = TF.normalize(img, self.mean, self.std)
        s = {'state': None}
        return img, s