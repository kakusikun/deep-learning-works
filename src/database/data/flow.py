from src.database.data import *
import os.path as osp
import h5py
import cv2
from scipy.io import loadmat
from src.database.data.market1501 import Market1501, Market1501LMDB

class FLOW(Market1501):
    """
    using market1501 data arrangement
    
    Dataset statistics:
    # identities: 971
    # images: 3884 (train) + 0 (query) + 0 (gallery)
    """
    
    def __init__(self, path="", branch="", use_train=False, use_test=False, **kwargs):
        super().__init__(path=path, branch=branch, use_train=use_train, use_test=use_test)
