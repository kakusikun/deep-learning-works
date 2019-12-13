from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
logger = logging.getLogger("logger")


class BaseData():
    handle = {}
    index_map = {}
    n_samples = {}


# import os
# import glob
# import re
# import sys
# import urllib 
# import tarfile
# import zipfile
# import os.path as osp
# from scipy.io import loadmat
# import numpy as np
# import h5py
# import cv2
# import pandas as pd
# from collections import defaultdict
# from tqdm import tqdm


# from tools.utils import mkdir_if_missing, write_json, read_json