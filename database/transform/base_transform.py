from abc import ABC, abstractmethod

class BaseTransform(ABC):
    def __init__(self):
        pass
    
    @classmethod
    def __repr__(cls):
        return cls.__name__

    @abstractmethod
    def apply_image(self, img):
        pass
    
    def apply_bbox(self, bbox, s):
        return bbox
    
    def apply_pts(self, cid, pts, s):
        return pts