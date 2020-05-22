import numpy as np
import cv2
import torchvision.transforms.functional as TF
import random
from PIL import Image
from tools.image import get_affine_transform, affine_transform
from src.database.transform import *

class RandScale(BaseTransform):
    '''
    To randomly move image on an area of final output size with random scale and without distortion

    Args:
        size (tuple): the output size

    Attributes:
        size (tuple): arg, size
    '''

    def __init__(self, size):
        self.size = size
        self.op_name = 'Scale'
    
    def apply_image(self, img):
        '''
        Resize image with random scale and position
        Args:
            img (PIL image): image to be resized
        Return:
            img (PIL image): resized image
            s (dict):
                ratio (tuple), scale of width and height
        '''

        np_img = np.array(img)
        h, w = np_img.shape[0], np_img.shape[1] 
        in_w, in_h = self.size
        c = np.array([w / 2., h / 2.], dtype=np.float32)
        s = max(h, w) * 1.0

        # image is randomly dragged to an center area with width and height of half of original
        w_border = get_border(int(w * 0.4), w)
        h_border = get_border(int(h * 0.4), h)
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        c[0] = np.random.randint(low=w_border, high=w - w_border)
        c[1] = np.random.randint(low=h_border, high=h - h_border)            

        trans_input = get_affine_transform(c, s, 0, [in_w, in_h])
        np_img = cv2.warpAffine(np_img, trans_input, (in_w, in_h), flags=cv2.INTER_LINEAR)
        img = Image.fromarray(np_img)
        s = {'c': c, 's': s}
        return img, s
    
    def apply_bbox(self, bbox, s):
        '''
        Resize bbox with random scale and position
        Args:
            bbox (numpy.ndarray, shape 1x4): bbox to be resized
            s (dict):
                ratio (tuple), scale of width and height recorded in function, apply_image
        Return:
            bbox (numpy.ndarray, shape 1x4): resized bbox
        '''

        assert 'c' in s
        assert 's' in s
        out_w, out_h = np.array(self.size)
        trans_output = get_affine_transform(s['c'], s['s'], 0, [out_w, out_h])
        bbox[:2] = affine_transform(bbox[:2], trans_output)
        bbox[2:] = affine_transform(bbox[2:], trans_output)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, out_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, out_h - 1) 
        return bbox

    def apply_pts(self, cid, pts, s):
        '''
        Resize keypoints with random scale and position
        Args:
            cid (int): the class for keypoints
            pts (numpy.ndarray, shape Nx2): keypoints to be resized
            s (dict):
                ratio (tuple), scale of width and height recorded in function, apply_image
        Return:
            pts (numpy.ndarray, shape Nx2): resized keypoints
        '''
        assert 'c' in s
        assert 's' in s
        out_w, out_h = np.array(self.size)
        trans_output = get_affine_transform(s['c'], s['s'], 0, [out_w, out_h])
        for i in range(pts.shape[0]):
            pts[i,:2] = affine_transform(pts[i,:2], trans_output)
            if ((pts[i, :2] < 0).sum() + (pts[i, :2] > (out_w, out_h)).sum()) > 0:
                pts[i, 2] = 0.0

        return pts


def get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i
