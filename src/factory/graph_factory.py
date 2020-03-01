from src.graph.simple_classifier import SimpleClassifier

class GraphFactory():
    products = {
        'simple_classifier': SimpleClassifier,
    }

    @classmethod
    def get_products(cls):
        return list(cls.products.keys())

    @classmethod
    def produce(cls, cfg, graph=None):
        if cfg.GRAPH not in cls.products:
            raise KeyError
        else:
            return cls.products[cfg.GRAPH if graph is None else graph]()
