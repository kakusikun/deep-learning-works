import torch
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image
from src.database.transform import *

class RandomPadding(BaseTransform):
    """Random rotate
    """

    def __init__(self, p=0.5, padding=(0, 10), **kwargs):
        self.p = p
        self.padding_limits = padding

    def apply_image(self, img, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return img
        rnd_padding = [random.randint(self.padding_limits[0], self.padding_limits[1]) for _ in range(4)]
        rnd_fill = random.randint(0, 255)
        return F.pad(img, tuple(rnd_padding), fill=rnd_fill, padding_mode='constant')

   