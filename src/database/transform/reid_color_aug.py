import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from abc import ABC, abstractmethod
from PIL import Image, ImageOps, ImageEnhance


class BaseTransform(ABC):

    def __init__(self, mag=0.0, prob=0.5):
        self.prob = prob
        self.mag = mag

    def __call__(self, img):
        return transforms.RandomApply([self.transform], self.prob)(img)

    def __repr__(self):
        return '%s(prob=%.2f, magnitude=%.2f)' % \
                (self.__class__.__name__, self.prob, self.mag)

    @abstractmethod
    def transform(self, img):
        pass


class ShearXY(BaseTransform):

    def transform(self, img):
        degrees = self.mag * 360
        t = transforms.RandomAffine(0, shear=degrees, resample=Image.BILINEAR)
        return t(img)

class AutoContrast(BaseTransform):

    def transform(self, img):
        cutoff = np.random.randint(0, 50, 1)[0]
        return ImageOps.autocontrast(img, cutoff=cutoff)

class Equalize(BaseTransform):

    def transform(self, img):
        return ImageOps.equalize(img)

class RandomBrightness(BaseTransform):

    def transform(self, img):
        factor = (np.random.rand(1)[0]/2) + 0.5
        img = ImageOps.equalize(img)
        return ImageEnhance.Brightness(img).enhance(factor)

class RandomVerticalClip(BaseTransform):

    def transform(self, img):
        img = np.array(img)
        h = img.shape[0]
        clip_pt = np.random.randint(h/self.mag, h, 1)[0]
        img = Image.fromarray(img[:clip_pt, ...])        
        return img

class REID_Color_aug(BaseTransform):
    DEFALUT_CANDIDATES = [
        Equalize(prob=1),
        RandomBrightness(),
        AutoContrast(),
        RandomVerticalClip(mag=3)
    ]
    def transform(self, img):
        for trans in self.DEFALUT_CANDIDATES:
            img = trans(img)
        return img