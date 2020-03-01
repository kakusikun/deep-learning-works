from src.model.backbone.osnet import osnet
from src.model.backbone.shufflenet import shufflenetv2_plus

class BackboneFactory:
    products = {
        'osnet': osnet,
        'shufflenetv2+': shufflenetv2_plus,
    }

    @classmethod
    def get_products(cls):
        return list(cls.products.keys())

    @classmethod
    def produce(cls, cfg, name=None):
        if cfg.MODEL.BACKBONE not in cls.products:
            raise KeyError
        else:
            return cls.products[cfg.MODEL.BACKBONE if name is None else name](
                shufflenetv2_plus_model_size='Large',
            )
