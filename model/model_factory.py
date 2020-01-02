from model.large_hourglass import get_large_hourglass_net
from model.OSNetv2 import osnet_x1_0
from model.OSNet_PFPN import get_osnet_pfpn

model_factory = {
    'light_hourglass': get_large_hourglass_net,
    'osnet': osnet_x1_0,
    'osnet_pfpn': get_osnet_pfpn
}

def get_names():
    return list(model_factory.keys())

def get_model(name):
    if name not in model_factory.keys():
        raise KeyError("Invalid model, got '{}', but expected to be one of {}".format(name, model_factory.keys()))   
    return model_factory[name]