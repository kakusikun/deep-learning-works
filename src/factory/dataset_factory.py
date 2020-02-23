from src.base_factory import BaseFactory
from src.database.dataset.coco import build_coco_dataset
from src.database.dataset.reid import build_reid_dataset
from src.database.dataset.imagenet import build_image_dataset
from src.database.dataset.cifar10 import build_cifar_dataset

class DatasetFactory:
    products = {
        'coco': build_coco_dataset,
        'reid': build_reid_dataset,
        'imagenet': build_image_dataset,
        'cifar10': build_cifar_dataset,
    }

    @classmethod
    def get_products(cls):
        return list(cls.products.keys())

    @classmethod
    def produce(cls, cfg, data=None, transform=None, build_func=None,
                return_indice=False):
        if cfg.DB.DATA not in cls.products:
            raise KeyError
        else:
            return cls.products[cfg.DB.DATASET](
                        data=data,
                        transform=transform,
                        build_func=build_func,
                        return_indice=return_indice
                    )
