from engine.engines.engine_center import CenterEngine
from engine.engines.engine_center_kp import CenterKPEngine
from engine.engines.engine_imagenet import ImageNetEngine
from engine.engines.engine_par import PAREngine
from engine.engines.engine_reid_trick import ReIDEngine

engine_factory = {
    'center': CenterEngine,
    'center_kp': CenterKPEngine,
    'imagenet': ImageNetEngine,
    'par': PAREngine,
    'reid_trick': ReIDEngine
}

def get_names():
    return list(engine_factory.keys())

def get_engine(name):
    if name not in engine_factory.keys():
        raise KeyError("Invalid engine, got '{}', but expected to be one of {}".format(name, engine_factory.keys()))   
    return engine_factory[name]