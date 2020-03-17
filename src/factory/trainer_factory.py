from src.trainer.trainers import (
    CenternetODTrainer, 
    ImagenetTrainer,
    SPOSClassificationTrainer,
)

class TrainerFactory:
    products = {
        'imagenet': ImagenetTrainer,
        'centernet_object_detection': CenternetODTrainer,
        'spos_classification': SPOSClassificationTrainer,
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