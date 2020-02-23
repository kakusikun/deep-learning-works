from src.database.loader import *
from tools.centerface_utils import centerface_facial_target
from tools.centernet_utils import centernet_keypoints_target

def build_coco_loader(cfg):
    data = DataFactory.produce(cfg.DB.DATA)(cfg)
    if cfg.MODEL.HEAD == 'center_kp':
        build_func = centernet_keypoints_target
    elif cfg.MODEL.HEAD == 'centerface':
        build_func = centerface_facial_target
    else:
        build_func = None
    loader = {}
    if cfg.DB.USE_TRAIN:
        train_trans = TransformFactory.produce(cfg, cfg.TRAIN_TRANSFORM)
        train_dataset = DatasetFactory.produce(cfg.DB.DATASET)(data.train, train_trans, build_func=build_func)
        loader['train'] = DataLoader(train_dataset, 
                                          batch_size=cfg.INPUT.TRAIN_BS, 
                                          shuffle=True, 
                                          num_workers=cfg.NUM_WORKERS, 
                                          pin_memory=True,
                                          drop_last=True
                                )
        cfg.SOLVER.ITERATIONS_PER_EPOCH = len(loader['train'])
    if cfg.DB.USE_TEST:
        val_trans = TransformFactory.produce(cfg, cfg.TEST_TRANSFORM)
        val_dataset = DatasetFactory.produce(cfg.DB.DATASET)(data.val, val_trans)
        loader['val'] = DataLoader(val_dataset, 
                                        batch_size=cfg.INPUT.TEST_BS, 
                                        shuffle=False, 
                                        num_workers=cfg.NUM_WORKERS, 
                                        pin_memory=True, 
                                        drop_last=False
                                   ) 
    return loader
