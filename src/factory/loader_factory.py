from src.database.loader.coco import build_coco_loader
from src.database.loader.reid import build_reid_loader
from src.database.loader.imagenet import build_imagenet_loader
from src.database.loader.cifar10 import build_cifar10_loader

class LoaderFactory:
    '''
    To get the pytorch data loader.
    3 levels, data => dataset => loader
    
    data:
        First, data having two attributes (dict)
            1. train
            2. val
        each attribute indicates three info (keys)
            1. handle, the map between index and data
            2. n_samples, number of data
            3. indice, if necessary, the map between index and file path
    
    Second, use the data to build dataset 
    Third, use the dataset to build loader

    Args:
        name (str): the avaidable name for loader. coco, reid, imagenet, cifar10
    '''
    products = {
        'coco': build_coco_loader,
        'reid': build_reid_loader,
        'imagenet': build_imagenet_loader,
        'cifar10': build_cifar10_loader,
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
            return cls.products[cfg.DB.LOADER](
                        data=data,
                        transform=transform,
                        build_func=build_func,
                        return_indice=return_indice
                    )