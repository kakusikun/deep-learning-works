from src.engine.classification import ClassificationEngine
from src.engine.centernet_object_detection import CenternetODEngine
from src.engine.hourglass_object_detection import HourglassODEngine
from src.engine.spos_classification import SPOSClassificationEngine
from src.engine.trick_reid import TrickReIDEngine
from src.engine.harmattn_reid import HarmAttnReIDEngine
from src.engine.iap_reid import IAPReIDEngine
from src.engine.hourglass_jde import HourglassJDE
from src.engine.shufflenetv2_jde import Shufflenetv2JDE

class EngineFactory():
    products = {
        'classification': ClassificationEngine,
        'centernet_object_detection': CenternetODEngine,
        'hourglass_object_detection': HourglassODEngine,
        'spos_classification': SPOSClassificationEngine,
        'trick_reid': TrickReIDEngine,
        'ha_reid': HarmAttnReIDEngine,
        'iap_reid': IAPReIDEngine,
        'hourglass_jde': HourglassJDE,
        'shufflenetv2_jde': Shufflenetv2JDE,
    }

    @classmethod
    def get_products(cls):
        return list(cls.products.keys())

    @classmethod
    def produce(cls, cfg, graph, loader, solvers, visualizer):
        if cfg.ENGINE not in cls.products:
            raise KeyError
        else:
            return cls.products[cfg.ENGINE](
                cfg=cfg, 
                graph=graph,
                loader=loader,
                solvers=solvers,
                visualizer=visualizer)
