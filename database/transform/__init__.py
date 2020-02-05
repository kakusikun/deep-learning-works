from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from database.transform.randaugment import RandAugment
from database.transform.normalize import Normalize
from database.transform.random_hflip import RandomHFlip
from database.transform.resize import Resize
from database.transform.resize_keep_aspect_ratio import ResizeKeepAspectRatio
from database.transform.tensorize import Tensorize
from database.transform.random_scale import RandScale
from database.transform.augmix import AugMix
from database.transform.random_crop import RandCrop