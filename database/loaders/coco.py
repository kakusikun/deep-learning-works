from database.loaders import *

def build_coco_loader(cfg):
    data = get_data(cfg.DB.DATA)(cfg)
    loader = {}
    if cfg.DB.USE_TRAIN:
        train_dataset = get_dataset(cfg.DB.DATASET)(data.train, data.index_map['train'], split='train', output_stride=cfg.MODEL.STRIDE)
        loader['train'] = Data.DataLoader(train_dataset, 
                                          batch_size=cfg.INPUT.TRAIN_BS, 
                                          shuffle=True, 
                                          num_workers=cfg.NUM_WORKERS, 
                                          pin_memory=True,
                                          drop_last=True
                                )
        cfg.SOLVER.ITERATIONS_PER_EPOCH = len(loader['train'])
    if cfg.DB.USE_TEST:
        val_dataset = get_dataset(cfg.DB.DATASET)(data.val, data.index_map['val'], split='val', output_stride=cfg.MODEL.STRIDE)
        loader['val'] = Data.DataLoader(val_dataset, 
                                        batch_size=cfg.INPUT.TEST_BS, 
                                        shuffle=False, 
                                        num_workers=cfg.NUM_WORKERS, 
                                        pin_memory=True, 
                                        drop_last=False
                                   ) 
    return loader
