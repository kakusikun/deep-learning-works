from database.datasets.coco import build_coco_dataset
from database.datasets.reid import build_reid_dataset
from database.datasets.imagenet import build_image_dataset
from database.datasets.cifar10 import build_cifar_dataset


dataset_factory = {
    'coco': build_coco_dataset,
    'reid': build_reid_dataset,
    'imagenet': build_image_dataset,
    'cifar10': build_cifar_dataset,
}

def get_names():
    return list(dataset_factory.keys())

def get_dataset(name):
    if name not in dataset_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, dataset_factory.keys()))   
    return dataset_factory[name]
