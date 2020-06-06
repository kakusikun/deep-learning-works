from src.graph.simple_classifier import SimpleClassifier
from src.graph.centernet_object_detection import CenterNetObjectDetection
from src.graph.shufflenetv2plus_classification import ShuffleNetV2PlusClassifier
from src.graph.hourglass_object_detection import HourglassObjectDetection
from src.graph.shufflenetv2_spos import ShuffleNetv2SPOS
from src.graph.trick_reid import TrickReID
from src.graph.harmattn_reid import HarmAttenReID
from src.graph.iap_reid import IAPReID, DualNormIAPReID
from src.graph.hourglass_jde import HourglassJDE
from src.graph.shufflenetv2_jde import ShuffleNetv2JDE
from src.graph.shufflenetv2_object_detection import ShuffleNetv2OD
from src.graph.shufflenetv2_scopehead_object_detection import ShuffleNetv2ScopeHeadOD
from src.graph.shufflenetv2_plus_csp import ShuffleNetv2CSP

class GraphFactory:
    products = {
        'simple_classifier': SimpleClassifier,
        'centernet_object_detection': CenterNetObjectDetection,
        'shufflenetv2plus_classification': ShuffleNetV2PlusClassifier,
        'hourglass_object_detection': HourglassObjectDetection,
        'shufflenetv2_spos_classification': ShuffleNetv2SPOS,
        'trick_reid': TrickReID,
        'ha_reid': HarmAttenReID,
        'iap_reid': IAPReID,
        'hourglass_jde': HourglassJDE,
        'shufflenetv2_jde': ShuffleNetv2JDE,
        'shufflenetv2_object_detection': ShuffleNetv2OD,
        'dualnorm_iap_reid': DualNormIAPReID,
        'shufflenetv2_scopehead_object_detection': ShuffleNetv2ScopeHeadOD,
        'shufflenetv2_csp': ShuffleNetv2CSP
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
