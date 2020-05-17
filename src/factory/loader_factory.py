from src.database.loader.coco import build_coco_loader
from src.database.loader.reid import build_reid_loader
from src.database.loader.classification import build_classification_loader
import torch.distributed as dist

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
        'classification': build_classification_loader,
    }

    @classmethod
    def get_products(cls):
        return list(cls.products.keys())

    @classmethod
    def produce(cls, cfg, loader_name=None):
        if cfg.DB.LOADER not in cls.products:
            raise KeyError
        else:
            loader = cls.products[cfg.DB.LOADER if loader_name is None else loader_name](
                        cfg, 
                        target_format=cfg.DB.TARGET_FORMAT,
                        use_train=cfg.DB.USE_TRAIN, 
                        use_test=cfg.DB.USE_TEST,
                        train_transformation=cfg.DB.TRAIN_TRANSFORM, 
                        test_transformation=cfg.DB.TEST_TRANSFORM,
                        train_batch_size=cfg.INPUT.TRAIN_BS, 
                        test_batch_size=cfg.INPUT.TEST_BS,
                        num_workers=cfg.NUM_WORKERS,
                        num_people_per_batch=cfg.REID.SIZE_PERSON,
                    )
            if cfg.DB.USE_TRAIN:
                cfg.SOLVER.ITERATIONS_PER_EPOCH = len(loader['train'])
            return loader