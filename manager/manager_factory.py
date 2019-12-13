from manager.managers.manager_center import CenterManager
from manager.managers.manager_center_kp import CenterKPManager
from manager.managers.manager_imagenet import ImageNetManager
from manager.managers.manager_par import PARManager
from manager.managers.manager_reid_trick import TrickManager

manager_factory = {
    'center': CenterManager,
    'center_kp': CenterKPManager,
    'imagenet': ImageNetManager,
    'par': PARManager,
    'reid_trick': TrickManager,
}

def get_names():
    return list(manager_factory.keys())

def get_manager(name):
    if name not in manager_factory.keys():
        raise KeyError("Invalid manager, got '{}', but expected to be one of {}".format(name, manager_factory.keys()))   
    return manager_factory[name]