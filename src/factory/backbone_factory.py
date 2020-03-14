from src.model.backbone.osnet import osnet
from src.model.backbone.shufflenetv2_plus import shufflenetv2_plus
from src.model.backbone.hourglass import hourglass_net
class BackboneFactory:
    products = {
        'osnet': osnet,
        'shufflenetv2+': shufflenetv2_plus,
        'hourglass': hourglass_net,
    }

    @classmethod
    def get_products(cls):
        return list(cls.products.keys())

    @classmethod
    def produce(cls, cfg, name=None):
        if cfg.MODEL.BACKBONE not in cls.products:
            raise KeyError
        else:
            return cls.products[cfg.MODEL.BACKBONE if name is None else name]()
