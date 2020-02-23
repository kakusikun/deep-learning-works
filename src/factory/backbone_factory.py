from src.base_factory import BaseFactory
from src.model.backbone import osnet

class BackboneFactory(BaseFactory):
    products = {
        'osnet': osnet
    }