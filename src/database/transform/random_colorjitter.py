import torch
import numpy as np
import torchvision.transforms as T
import random
from PIL import Image
from src.database.transform import *

class RandomColorJitter(BaseTransform):
    def __init__(self, p=0.5, brightness=0.2, contrast=0.15, saturation=0, hue=0, **kwargs):
        self.p = p
        self.color_jitter = T.ColorJitter(
            brightness=brightness, 
            contrast=contrast, 
            saturation=saturation, 
            hue=hue
        )


    def apply_image(self, img):
        if random.uniform(0, 1) > self.p:
            return img
        transform = self.color_jitter.get_params(
            self.color_jitter.brightness, 
            self.color_jitter.contrast,
            self.color_jitter.saturation, 
            self.color_jitter.hue)
        return transform(img)

   