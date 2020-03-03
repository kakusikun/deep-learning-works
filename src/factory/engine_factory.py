from src.engine.classification import ClassificationEngine
from src.engine.centernet_object_detection import CenternetODEngine

class EngineFactory():
    products = {
        'classification': ClassificationEngine,
        'centernet_object_detection': CenternetODEngine,
    }

    @classmethod
    def get_products(cls):
        return list(cls.products.keys())

    @classmethod
    def produce(cls, cfg, graph, loader, solvers, visualizer):
        if cfg.GRAPH not in cls.products:
            raise KeyError
        else:
            return cls.products[cfg.GRAPH if graph is None else graph](
                cfg=cfg, 
                graph=graph,
                loader=loader,
                solvers=solvers,
                visualizer=visualizer)
