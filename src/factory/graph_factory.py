from src.base_factory import BaseFactory
from src.graph.classifier import Classifier

class GraphFactory(BaseFactory):
    products = {
        'classifier': Classifier,
    }
