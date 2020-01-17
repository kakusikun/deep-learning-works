import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image

PERSON_KEYPOINT_INDICE = {0:[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]}
PERSON_KEYPOINT_INDICE_OFFSET = [0]

CLOTHE_KEYPOINT_INDICE = {0:[[1,5],[2,4],[6,24],[7,23],[8,22],[9,21],[10,20],[11,19],[12,18],[13,17],[14,16]],
                          1:[[1,5],[2,4],[6,32],[7,31],[8,30],[9,29],[10,28],[11,27],[12,26],[13,25],[14,24],[15,23],[16,22],[17,21],[18,20]],
                          2:[[1,25],[2,4],[3,5],[6,24],[7,23],[8,22],[9,21],[10,20],[11,19],[12,18],[13,17],[14,16],[15,28],[30,27],[29,26]],
                          3:[[1,5],[2,4],[3,33],[6,32],[7,31],[8,30],[9,29],[10,28],[11,27],[12,26],[13,25],[14,24],[15,23],[16,22],[17,21],[18,20],[19,36],[38,35],[37,34]],
                          4:[[1,5],[2,4],[6,14],[7,13],[8,12],[9,11]],
                          5:[[1,5],[2,4],[6,14],[7,13],[8,12],[9,11]],
                          6:[[0,2],[3,9],[4,8],[5,7]],
                          7:[[0,2],[3,13],[4,12],[5,11],[6,10],[7,9]],
                          8:[[0,2],[3,7],[4,6]],
                          9:[[1,5],[2,4],[6,28],[7,27],[8,26],[9,25],[10,24],[11,23],[12,22],[13,21],[14,20],[15,19],[16,18]],
                          10:[[1,5],[2,4],[6,36],[7,35],[8,34],[9,33],[10,32],[11,31],[12,30],[13,29],[14,28],[15,27],[16,26],[17,25],[18,24],[19,23],[20,22]],
                          11:[[1,5],[2,4],[6,18],[7,17],[8,16],[9,15],[10,14],[11,13]],
                          12:[[1,5],[2,4],[6,18],[7,17],[8,16],[9,15],[10,14],[11,13]]}
CLOTHE_KEYPOINT_INDICE_OFFSET = [0,25,58,89,128,143,158,168,182,190,219,256,275]

class RandomHFlip():
    '''
    To flip the image horizontally
    Args:
        num_keypoints (int): optional, number of category of keypoints. 
                          e.g., in COCO dataset, person has 17 types of keypoints.
                          For keypoint detection, not noly the position of 
                          keypoints changes but the category changes
    '''
    def __init__(self, num_keypoints=0):
        if num_keypoints > 0:
            if num_keypoints == 17:
                self.flip_idx = PERSON_KEYPOINT_INDICE
                self.flip_idx_offset = PERSON_KEYPOINT_INDICE_OFFSET
            else:
                self.flip_idx = CLOTHE_KEYPOINT_INDICE
                self.flip_idx_offset = CLOTHE_KEYPOINT_INDICE_OFFSET
        else:
            self.flip_idx = []
            self.flip_idx_offset = {}

    def apply_image(self, img):
        '''
        Flip image
        Args:
            img (PIL image): image to be resized
        Return:
            img (PIL image): resized image
            s (dict):  
                flipped (bool), whether the image is flipped,
                w (int), the width of image
        '''
        flipped = False
        w, h = img.size
        if random.random() > 0.5:
            img = TF.hflip(img)
            flipped = True
        state = {'flipped': flipped, 'w': img.size[0]}
        return img, state
    
    def apply_bbox(self, bbox, s):
        '''
        Flip bbox
        Args:
<<<<<<< HEAD
            bbox (numpy.ndarray, shape 1x4): bbox to be resized
=======
            bbox (numpy array, shape 1x4): bbox to be resized
>>>>>>> e93d3bef6882492b37dee831975daf57cad341f1
            s (dict):  
                flipped (bool), whether the image is flipped,
                w (int), the width of image,
                recorded in function, apply_image
        Return:
<<<<<<< HEAD
            bbox (numpy.ndarray, shape 1x4): flipped bbox
=======
            bbox (numpy array, shape 1x4): flipped bbox
>>>>>>> e93d3bef6882492b37dee831975daf57cad341f1
        '''
        assert 'flipped' in s
        assert 'w' in s
        if s['flipped']:
            bbox[[0, 2]] = s['w'] - bbox[[2, 0]] - 1
        return bbox
    
    def apply_pts(self, cid, pts, s):
        '''
        Flip keypoints
        Args:
            cid (int): the class for keypoints
<<<<<<< HEAD
            pts (numpy.ndarray, shape Nx2): keypoints to be resized
=======
            pts (numpy array, shape Nx2): keypoints to be resized
>>>>>>> e93d3bef6882492b37dee831975daf57cad341f1
            s (dict):  
                flipped (bool), whether the image is flipped,
                w (int), the width of image,
                recorded in function, apply_image
        Return:
<<<<<<< HEAD
            pts (numpy.ndarray, shape Nx2): flipped keypoints
=======
            pts (numpy array, shape Nx2): flipped keypoints
>>>>>>> e93d3bef6882492b37dee831975daf57cad341f1
        '''
        assert 'flipped' in s
        assert 'w' in s
        assert len(self.flip_idx) > 0

        if s['flipped']:
            pts[:, 0] = s['w'] - pts[:, 0] - 1
            for e in self.flip_idx[cid]:
                e_offset = self.flip_idx_offset[cid]
                pts[e[0]+e_offset], pts[e[1]+e_offset] = pts[e[1]+e_offset].copy(), pts[e[0]+e_offset].copy()
        return pts

