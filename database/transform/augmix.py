import random

import numpy as np
import math
from database.transform.base_transform import BaseTransform
from database.transform import augmentations as aug

class AugMix(BaseTransform):
    '''
    Perform AugMix augmentations and compute mixture (https://arxiv.org/abs/1912.02781).
    Reference : 
        AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty
        https://github.com/google-research/augmix
                
    Args:
        width (int): optional, number of augmentation chains applied on image, default is 3
        depth (int): optional, length of augmentations to form a augmentation chain, 
                     default is -1 which uses random length in [1, 3]
        mag (int): optional, the severities of augmentation, default is 3 which uses random
                   severities in [0.1, 3]
    '''

    def __init__(self, width=3, depth=-1, mag=3):
        self.width = width
        self.depth = depth+1 if depth > 0 else 4
        self.mag = mag

    def apply_image(self, img):   
        '''
        Args:
            image (PIL.Image): input image
        Returns:
            mixed (numpy.ndarray): Augmented and mixed image.
        '''

        ws = np.random.dirichlet([1] * self.width).astype(np.float32)
        m = np.random.beta(1, 1)

        mix = np.zeros_like(np.array(img), dtype=np.float32)
        for i in range(self.width):
            depth = np.random.randint(1, self.depth)
            for _ in range(depth):
                img_aug = img.copy()                
                op_name = np.random.choice(aug.AUGMIX_OPS_NAME)
                mag = np.random.uniform(low=0.1, high=self.mag)
                level = aug.AUG_LEVELS[op_name](mag, size = img.size)
                img_aug = aug.AUG_OPS[op_name](img_aug, level)
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * np.array(img_aug).astype(np.float32)

        mixed = (1 - m) * np.array(img).astype(np.float32) + m * mix
        s = {'state': None}        
        return mixed, s


