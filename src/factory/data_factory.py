from src.database.data.coco import coco_data
from src.database.data.market1501 import Market1501
from src.database.data.cuhk import CUHK01, CUHK02, CUHK03
from src.database.data.dukemtmc import DukeMTMCreID
from src.database.data.msmt import MSMT17
from src.database.data.imagenet import ImageNet
from src.database.data.cifar10 import Cifar10

class DataFactory:
    products = {
        'coco': coco_data,
        'coco_person_kp': coco_data,
        'deepfashion': coco_data,
        'market': Market1501,
        'cuhk01': CUHK01,
        'cuhk02': CUHK02,
        'cuhk03': CUHK03,
        'duke': DukeMTMCreID,
        'msmt': MSMT17,
        'imagenet': ImageNet,
        'cifar10': Cifar10,
        'widerface': coco_data,
    }

    @classmethod
    def get_products(cls):
        return list(cls.products.keys())

    @classmethod
    def produce(cls, cfg, data_name=None):
        if cfg.DB.DATA not in cls.products:
            raise KeyError
        else:
            return cls.products[cfg.DB.DATA if data_name is None else data_name](
                        path=cfg.DB.PATH,
                        branch=cfg.DB.DATA if data_name is None else data_name,
                        coco_target=cfg.COCO.TARGET,
                        num_keypoints=cfg.DB.NUM_KEYPOINTS,
                        num_classes=cfg.DB.NUM_CLASSES,
                        output_stride=cfg.MODEL.STRIDE,
                        is_merge=cfg.REID.MERGE,
                        use_train=cfg.DB.USE_TRAIN,
                        use_test=cfg.DB.USE_TEST,
                    )

