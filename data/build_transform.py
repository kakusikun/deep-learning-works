import random
import math
import numbers
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
from data.transforms import *

def build_transform(cfg, =True):
    bagTransforms = []
    
<<<<<<< HEAD
    if :
        if cfg.TRANSFORM.RANDOMAPPLY:
            num_trans = np.random.randint(0, len(DEFALUT_CANDIDATES)+1, 1)
            trans = np.random.choice(DEFALUT_CANDIDATES, num_trans, replace=False)
            bagTransforms.extend(trans)
=======
    if isTrain:
        if cfg.TRANSFORM.AUGMENT:
            bagTransforms.append(ReID_Augment())
>>>>>>> mar_temp

        if cfg.TRANSFORM.RESIZE:
            bagTransforms.append(T.Resize(size=cfg.INPUT.IMAGE_SIZE))

        if cfg.TRANSFORM.HFLIP:
            bagTransforms.append(T.RandomHorizontalFlip(p=cfg.INPUT.PROB))
            
        if cfg.TRANSFORM.RANDOMCROP:
<<<<<<< HEAD
            bagTransforms.append(T.RandomCrop(size=cfg.INPUT.IMAGE_CROP_SIZE, padding=cfg.INPUT.IMAGE_PAD))          
        
=======
            bagTransforms.append(T.RandomCrop(size=cfg.INPUT.IMAGE_CROP_SIZE, padding=cfg.INPUT.IMAGE_PAD))   

>>>>>>> mar_temp
        bagTransforms.append(T.ToTensor())

        if cfg.TRANSFORM.NORMALIZE:
            bagTransforms.append(T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD))

        if cfg.TRANSFORM.RANDOMERASING:
            bagTransforms.append(RandomErasing())     
                   
    else:
        if cfg.TRANSFORM.SINGLE_CROP:
            if cfg.TRANSFORM.RESIZE:
                bagTransforms.append(T.Resize(size=cfg.INPUT.IMAGE_SIZE))
            bagTransforms.append(T.CenterCrop(size=cfg.INPUT.IMAGE_CROP_SIZE))
        else:            
            if cfg.TRANSFORM.RESIZE:
                bagTransforms.append(T.Resize(size=cfg.INPUT.IMAGE_CROP_SIZE))
        
        bagTransforms.append(T.ToTensor())
        if cfg.TRANSFORM.NORMALIZE:
            bagTransforms.append(T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD))
                
    transform = T.Compose(bagTransforms)

    return transform



class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img, -1, -1, -1, -1

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img, x1, y1, h, w

        return img, -1, -1, -1, -1
    
    def by_param(self, img, x1, y1, h, w):
        img[0, x1:x1 + h, y1:y1 + w] = 0
        img[1, x1:x1 + h, y1:y1 + w] = 0
        img[2, x1:x1 + h, y1:y1 + w] = 0
        return img

class _RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), i, j, h, w

    def by_param(self, img, i, j, h, w):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class _RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        is_flip = False
        if random.random() < self.p:
            is_flip = True
            return F.hflip(img), is_flip
        return img, is_flip

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

