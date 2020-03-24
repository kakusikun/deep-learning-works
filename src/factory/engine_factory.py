from src.engine.classification import ClassificationEngine
from src.engine.centernet_object_detection import CenternetODEngine
from src.engine.hourglass_object_detection import HourglassODEngine
from src.engine.spos_classification import SPOSClassificationEngine
from src.engine.trick_reid import TrickReIDEngine
from src.engine.harmattn_reid import HarmAttnReIDEngine

class EngineFactory():
    products = {
        'classification': ClassificationEngine,
        'centernet_object_detection': CenternetODEngine,
        'hourglass_object_detection': HourglassODEngine,
        'spos_classification': SPOSClassificationEngine,
        'trick_reid': TrickReIDEngine,
        'ha_reid': HarmAttnReIDEngine,
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
