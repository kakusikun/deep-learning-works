import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image
from src.database.transform import *

class ResizeFit(BaseTransform):
    '''
    To resize the image to be divisible

    Args:
        divisor (float): the number is divisible to image size

    Attributes:
        divisor (float): arg, divisor
    '''

    def __init__(self, divisor, long_side):
        self.divisor = divisor
        self.long_side = long_side
        self.op_name = 'ResizeFit'
    
    def apply_image(self, img):
        '''
        Resize image to be divisible
        Args:
            img (PIL image): image to be resized
        Return:
            img (PIL image): resized image
            s (dict):
                ratio (tuple), scale of width and height
        '''
        w, h = img.size
        if h > w:
            new_w = int(w/h * self.long_side)
            rest = new_w % self.divisor
            rest_w = self.divisor - rest if rest > self.divisor / 2 else -1 * rest
            r_w = (rest_w + new_w) / float(w)
            r_h = self.long_side / float(h)
            img = TF.resize(img, (self.long_side, rest_w + new_w))
        else:
            new_h = int(h/w * self.long_side)
            rest = new_h % self.divisor
            rest_h = self.divisor - rest if rest > self.divisor / 2 else -1 * rest
            r_w = self.long_side / float(w)
            r_h = (rest_h + new_h) / float(h)
            img = TF.resize(img, (rest_h + new_h, self.long_side))
        s = {'ratio': (r_w, r_h)}
        return img, s
    
    def apply_bbox(self, bbox, s):
        '''
        Resize bbox
        Args:
            bbox (numpy.ndarray, shape 1x4): bbox to be resized
            s (dict):
                ratio (tuple), scale of width and height recorded in function, apply_image
        Return:
            bbox (numpy.ndarray, shape 1x4): resized bbox
        '''
        assert 'ratio' in s
        r_w, r_h = s['ratio']
        bbox[[0, 2]] *= r_w
        bbox[[1, 3]] *= r_h
        return bbox

    def apply_pts(self, cid, pts, s):
        '''
        Resize keypoints
        Args:
            cid (int): the class for keypoints
            pts (numpy.ndarray, shape Nx2): keypoints to be resized
            s (dict):
                ratio (tuple), scale of width and height recorded in function, apply_image
        Return:
            pts (numpy.ndarray, shape Nx2): resized keypoints
        '''
        assert 'ratio' in s
        r_w, r_h = s['ratio']
        pts[:, 0] *= r_w
        pts[:, 1] *= r_h
        return pts