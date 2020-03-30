import torch
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image
from src.database.transform import *

class RandomGrayScale(BaseTransform):

    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def apply_image(self, img):
        s = {'state': None}
        if random.uniform(0, 1) > self.p:
            return img, s
        return TF.to_grayscale(img, num_output_channels=3), s
   