from trainer import CenterKPTrainer, CenterTrainer, ImagenetTrainer, ReIDTrainer

trainer_factory = {
    'center': CenterTrainer,
    'center_kp': CenterKPTrainer,
    'imagenet': ImagenetTrainer,
    'reid_trick': ReIDTrainer,
}

def get_names():
    return list(trainer_factory.keys())

def get_trainer(name):
    if name not in trainer_factory.keys():
        raise KeyError("Invalid trainer, got '{}', but expected to be one of {}".format(name, trainer_factory.keys()))   
    return trainer_factory[name]

