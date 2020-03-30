from src.trainer.trainers import (
    CenternetODTrainer, 
    ImagenetTrainer,
    SPOSClassificationTrainer,
    TrickReIDTrainer,
    HarmAttnReIDTrainer,
    IAPReIDTrainer,
)

class TrainerFactory:
    products = {
        'imagenet': ImagenetTrainer,
        'centernet_object_detection': CenternetODTrainer,
        'spos_classification': SPOSClassificationTrainer,
        'trick_reid': TrickReIDTrainer,
        'ha_reid': HarmAttnReIDTrainer,
        'iap_reid': IAPReIDTrainer,
    }

    @classmethod
    def get_products(cls):
        return list(cls.products.keys())

    @classmethod
    def produce(cls, cfg, trainer=None):
        if cfg.TRAINER not in cls.products:
            raise KeyError
        else:
            return cls.products[cfg.TRAINER if trainer is None else trainer](cfg)