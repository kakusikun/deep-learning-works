from src.base_factory import BaseFactory
from src.graph.simple_classifier import SimpleClassifier

class GraphFactory(BaseFactory):
    products = {
        'simple_classifier': SimpleClassifier,
    }
