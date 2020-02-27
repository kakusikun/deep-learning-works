from src.base_factory import BaseFactory
from src.model.backbone import osnet

#TODO: remove BaseFactory
class BackboneFactory(BaseFactory):
    products = {
        'osnet': osnet
    }

    @classmethod
    def get_products(cls):
        return list(cls.products.keys())

    @classmethod
    def produce(cls, cfg, name=None):
        if cfg.MODEL.BACKBONE not in cls.products:
            raise KeyError
        else:
            return cls.products[cfg.MODEL.BACKBONE]()
