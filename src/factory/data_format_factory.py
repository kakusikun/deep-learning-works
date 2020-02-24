from src.base_factory import BaseFactory
from src.database.data_format.coco import build_coco_dataset
from src.database.data_format.reid import build_reid_dataset
from src.database.data_format.imagenet import build_image_dataset
from src.database.data_format.cifar10 import build_cifar_dataset

class DataFormatFactory:
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
    def produce(cls, 
        cfg, 
        data=None, 
        data_format=None,
        transform=None, 
        build_func=None,
        return_indice=False):

        if cfg.DB.DATA_FORMAT not in cls.products:
            raise KeyError
        else:
            return cls.products[cfg.DB.DATA_FORMAT if data_format is None else data_format](
                        data=data,
                        transform=transform,
                        build_func=build_func,          # coco
                        return_indice=return_indice     # reid
                    )
