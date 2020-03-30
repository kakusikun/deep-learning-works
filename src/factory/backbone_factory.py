from src.model.backbone.shufflenetv2_plus import shufflenetv2_plus
from src.model.backbone.hourglass import hourglass_net
from src.model.backbone.shufflenetv2_torchvision import shufflenetv2, shufflenetv2_low_resolution
from src.model.backbone.osnet_deep_reid import osnet_x1_0
from src.model.backbone.hacnn import hacnn
from src.model.backbone.osnet_deep_reid_iap import osnet_iap_x1_0
class BackboneFactory:
    products = {
        'shufflenetv2+': shufflenetv2_plus,
        'hourglass': hourglass_net,
        'shufflenetv2': shufflenetv2,
        'shufflenetv2_low_resolution': shufflenetv2_low_resolution,
        'osnet_deep_reid': osnet_x1_0,
        'hacnn': hacnn,
        'osnet_deep_reid_iap': osnet_iap_x1_0,
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
