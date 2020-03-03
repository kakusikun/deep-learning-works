from src.graph.simple_classifier import SimpleClassifier
from src.graph.centernet_object_detection import CenterNetObjectDetection

class GraphFactory():
    products = {
        'simple_classifier': SimpleClassifier,
        'centernet_object_detection': CenterNetObjectDetection,
    }

    @classmethod
    def get_products(cls):
        return list(cls.products.keys())

    @classmethod
    def produce(cls, cfg, graph=None):
        if cfg.GRAPH not in cls.products:
            raise KeyError
        else:
            return cls.products[cfg.GRAPH if graph is None else graph](cfg)
