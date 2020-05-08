from src.database.data.coco import COCO
from src.database.data.market1501 import Market1501
from src.database.data.cuhk import CUHK01, CUHK02, CUHK03
from src.database.data.dukemtmc import DukeMTMCreID
from src.database.data.msmt import MSMT17
from src.database.data.imagenet import ImageNet
from src.database.data.cifar10 import Cifar10
from src.database.data.emotion import Emotion
from src.database.data.tinyimagenet import TinyImageNet
from src.database.data.flow import FLOW

class DataFactory:
    products = {
        'coco': COCO,
        'coco_person_kp': COCO,
        'deepfashion': COCO,
        'market': Market1501,
        'cuhk01': CUHK01,
        'cuhk02': CUHK02,
        'cuhk03': CUHK03,
        'duke': DukeMTMCreID,
        'msmt': MSMT17,
        'imagenet': ImageNet,
        'cifar10': Cifar10,
        'widerface': COCO,
        'emotion': Emotion,
        'tinyimagenet': TinyImageNet,
        'cuhksysu': COCO,
        'caltech': COCO,
        'cityperson': COCO,
        'ethz': COCO,
        'prw': COCO,
        'flow': FLOW,
        'crownhuman': COCO,
    }

    @classmethod
    def get_products(cls):
        return list(cls.products.keys())

    @classmethod
    def produce(cls, 
        cfg, 
        path=None,
        branch=None,
        coco_target=None,
        num_keypoints=None,
        num_classes=None,
        output_strides=None,
        use_all=None,
        use_train=None,
        use_test=None,
    ):
        if branch is None and cfg.DB.DATA not in cls.products:
            raise KeyError
        else:
            return cls.products[cfg.DB.DATA if branch is None else branch](
                        path=cfg.DB.PATH if path is None else path,
                        branch=cfg.DB.DATA if branch is None else branch,
                        coco_target=cfg.COCO.TARGET if coco_target is None else coco_target,
                        num_keypoints=cfg.DB.NUM_KEYPOINTS if num_keypoints is None else num_keypoints,
                        num_classes=cfg.DB.NUM_CLASSES if num_classes is None else num_classes,
                        output_strides=cfg.MODEL.STRIDES if output_strides is None else output_strides,
                        use_all=cfg.REID.MSMT_ALL if use_all is None else use_all,
                        use_train=cfg.DB.USE_TRAIN if use_train is None else use_train,
                        use_test=cfg.DB.USE_TEST if use_test is None else use_test,
                    )

