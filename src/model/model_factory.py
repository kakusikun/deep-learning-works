from model.large_hourglass import get_large_hourglass_net
from model.OSNet import osnet
from model.OSNet_PFPN import get_osnet_pfpn
from model.ResNet_cifar10 import ResNet18

model_factory = {
    'light_hourglass': get_large_hourglass_net,
    'osnet': osnet,
    'osnet_pfpn': get_osnet_pfpn,
    'resnet_cifar10': ResNet18,
}

def get_names():
    return list(model_factory.keys())

def get_model(name):
    if name not in model_factory.keys():
        raise KeyError("Invalid model, got '{}', but expected to be one of {}".format(name, model_factory.keys()))   
    return model_factory[name]