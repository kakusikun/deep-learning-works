from src.database.loader import *


def build_imagenet_loader(cfg):
    loader = {}
    data = DataFactory.produce(cfg.DB.DATA)(cfg)

    if cfg.DB.USE_TRAIN:
        train_trans = TransformFactory.produce(cfg, cfg.TRAIN_TRANSFORM)
        train_dataset = DatasetFactory.produce(cfg.DB.DATASET)(data.train, train_trans)
        loader['train'] = DataLoader(train_dataset, 
                                          batch_size=cfg.INPUT.TRAIN_BS, 
                                          shuffle=True, 
                                          num_workers=cfg.NUM_WORKERS, 
                                          pin_memory=True,
                                          drop_last=True)
        cfg.SOLVER.ITERATIONS_PER_EPOCH = len(loader['train'])
    if cfg.DB.USE_TEST:
        val_trans = TransformFactory.produce(cfg, cfg.TEST_TRANSFORM)
        val_dataset = DatasetFactory.produce(cfg.DB.DATASET)(data.val, val_trans)
        loader['val'] = DataLoader(val_dataset, 
                                        batch_size=cfg.INPUT.TEST_BS, 
                                        shuffle=False, 
                                        num_workers=cfg.NUM_WORKERS, 
                                        pin_memory=True,
                                        drop_last=False)

    return loader