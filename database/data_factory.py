from database.data.coco import coco_data
from database.data.market1501 import Market1501
from database.data.cuhk import CUHK01, CUHK02, CUHK03
from database.data.dukemtmc import DukeMTMCreID
from database.data.msmt import MSMT17
from database.data.imagenet import ImageNet
from database.data.cifar10 import Cifar10


data_factory ={
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

def get_names():
    return list(data_factory.keys())

def get_data(name):
    if name not in data_factory.keys():
        raise KeyError("Invalid data, got '{}', but expected to be one of {}".format(name, data_factory.keys()))
    return data_factory[name]