from database.loaders.coco import build_coco_loader
from database.loaders.reid import build_reid_loader
from database.loaders.imagenet import build_imagenet_loader

loader_factory = {
    'coco': build_coco_loader,
    'reid': build_reid_loader,
    'imagenet': build_imagenet_loader
}

def get_names():
    return list(loader_factory.keys())

def get_loader(name):
    if name not in loader_factory.keys():
        raise KeyError("Invalid loader, got '{}', but expected to be one of {}".format(name, loader_factory.keys()))   
    return loader_factory[name]
