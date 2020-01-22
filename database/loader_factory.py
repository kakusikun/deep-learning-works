from database.loaders.coco import build_coco_loader
from database.loaders.reid import build_reid_loader
from database.loaders.imagenet import build_imagenet_loader
from database.loaders.cifar10 import build_cifar10_loader

loader_factory = {
    'coco': build_coco_loader,
    'reid': build_reid_loader,
    'imagenet': build_imagenet_loader,
    'cifar10': build_cifar10_loader,
}

def get_names():
    return list(loader_factory.keys())

def get_loader(name):
    '''
    To get the pytorch data loader.
    First, get the data (database.data.BaseData) having two attributes (dict)
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
    if name not in loader_factory.keys():
        raise KeyError("Invalid loader, got '{}', but expected to be one of {}".format(name, loader_factory.keys()))   
    return loader_factory[name]
