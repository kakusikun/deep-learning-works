from database.loaders import *
import os.path as osp

def build_cifar10_loader(cfg):
    data = get_data(cfg.DB.DATA)(cfg)
    loader = {}
    if cfg.DB.USE_TRAIN:
        train_trans = get_transform(cfg, cfg.TRAIN_TRANSFORM)
        train_dataset = get_dataset(cfg.DB.DATASET)(data.handle['train'], train_trans)
        loader['train'] = Data.DataLoader(train_dataset, 
                                          batch_size=cfg.INPUT.TRAIN_BS, 
                                          shuffle=True, 
                                          num_workers=cfg.NUM_WORKERS, 
                                          pin_memory=True,
                                          drop_last=True)
        cfg.SOLVER.ITERATIONS_PER_EPOCH = len(loader['train'])
    if cfg.DB.USE_TEST:
        val_trans = get_transform(cfg, cfg.TEST_TRANSFORM)
        val_dataset = get_dataset(cfg.DB.DATASET)(data.handle['val'], val_trans)
        loader['val'] = Data.DataLoader(val_dataset, 
                                        batch_size=cfg.INPUT.TEST_BS, 
                                        shuffle=False, 
                                        num_workers=cfg.NUM_WORKERS, 
                                        pin_memory=True,
                                        drop_last=False)

    return loader