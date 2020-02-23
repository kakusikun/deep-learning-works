from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.database.transform.randaugment import RandAugment
from src.database.transform.normalize import Normalize
from src.database.transform.random_hflip import RandomHFlip
from src.database.transform.resize import Resize
from src.database.transform.resize_keep_aspect_ratio import ResizeKeepAspectRatio
from src.database.transform.tensorize import Tensorize
from src.database.transform.random_scale import RandScale
from src.database.transform.augmix import AugMix
from src.database.transform.random_crop import RandCrop
from src.database.transform.cutout import Cutout