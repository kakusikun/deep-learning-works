import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image

class Normalize():
    '''
    To normalize the data

    Args:
        mean (list): the value substracted from data
        std (list): the value divided from data

    Attributes:
        mean (list): arg, mean
        std (list): arg, std
    '''

    def __init__(self, size: tuple, stride=1):
        self.size = size
        self.stride = stride
    
    def apply_image(self, img):
        '''
        Resize image
        Args:
            img (PIL image): image to be resized
        Return:
            img (PIL image): resized image
            s (dict):
                ratio (tuple), scale of width and height
        '''
        w, h = img.size
        r_w = self.size[0] / float(w)
        r_h = self.size[1] / float(h)
        img = TF.resize(img, self.size)
        s = {'ratio': (r_w, r_h)}
        return img, s
    
    def apply_bbox(self, bbox, s):
        '''
        Resize bbox
        Args:
            bbox (numpy array, shape 1x4): bbox to be resized
            s (dict):
                ratio (tuple), scale of width and height recorded in function, apply_image
        Return:
            bbox (numpy array, shape 1x4): resized bbox
        '''
        assert 'ratio' in s
        r_w, r_h = s['ratio']
        r_w /= self.stride
        r_h /= self.stride
        bbox[[0, 2]] *= r_w
        bbox[[1, 3]] *= r_h
        return bbox

    def apply_pts(self, cid, pts, s):
        '''
        Resize keypoints
        Args:
            cid (int): the class for keypoints
            pts (numpy array, shape Nx2): keypoints to be resized
            s (dict):
                ratio (tuple), scale of width and height recorded in function, apply_image
        Return:
            pts (numpy array, shape Nx2): resized keypoints
        '''
        assert 'ratio' in s
        r_w, r_h = s['ratio']
        pts[:, 0] *= r_w
        pts[:, 1] *= r_h
        return pts