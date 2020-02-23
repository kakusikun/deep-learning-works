from src.database.loader import *
from src.database.sampler.sampler import IdBasedSampler

def build_reid_loader(cfg, return_indice=False, use_sampler=True):
    loader = {}
    val_trans = TransformFactory.produce(cfg, cfg.TEST_TRANSFORM)
    train_trans = TransformFactory.produce(cfg, cfg.TRAIN_TRANSFORM)

    if "merge" in cfg.DB.DATA:
        data_names = cfg.DB.DATA.split(" ")[1:]
        if cfg.DB.USE_TRAIN:   
            _data = [] 
            offset = 0
            for name in data_names[:-1]:
                cfg.DB.DATA = name
                data = DataFactory.produce(name)(cfg)
                cfg.DB.NUM_CLASSES += data.train['n_samples']
                for path, pid, cam in data.train['indice']: 
                    _data.append([path, pid+offset, cam])
                offset += data.train['n_samples']
            data.train['indice'] = _data
            data.train['n_samples'] = offset
            
            train_dataset = DatasetFactory.produce(cfg.DB.DATASET)(data.train, train_trans)
            if use_sampler:
                sampler = IdBasedSampler(data.train['indice'], batch_size=cfg.INPUT.TRAIN_BS, num_instances=cfg.REID.SIZE_PERSON)
        if cfg.DB.USE_TEST:
            data = DataFactory.produce(data_names[-1])(cfg)            
            query_dataset = DatasetFactory.produce(cfg.DB.DATASET)(data.query, val_trans)
            gallery_dataset = DatasetFactory.produce(cfg.DB.DATASET)(data.gallery, val_trans)
    else:
        data = DataFactory.produce(cfg.DB.DATA)(cfg)
        if cfg.DB.USE_TRAIN:
            train_dataset = DatasetFactory.produce(cfg.DB.DATASET)(data.train, train_trans)
            cfg.DB.NUM_CLASSES = data.n_samples['train']
            if use_sampler:
                sampler = IdBasedSampler(data.train['indice'], batch_size=cfg.INPUT.TRAIN_BS, num_instances=cfg.REID.SIZE_PERSON)
        if cfg.DB.USE_TEST: 
            query_dataset = DatasetFactory.produce(cfg.DB.DATASET)(data.query, val_trans)
            gallery_dataset = DatasetFactory.produce(cfg.DB.DATASET)(data.gallery, val_trans)

    if cfg.DB.USE_TRAIN: 
        if use_sampler:           
            loader['train'] = DataLoader(
                train_dataset, 
                batch_size=cfg.INPUT.TRAIN_BS, 
                sampler=sampler, 
                num_workers=cfg.NUM_WORKERS, 
                pin_memory=True, 
                drop_last=True
            )
        else:
            loader['train'] = DataLoader(
                train_dataset, 
                batch_size=cfg.INPUT.TRAIN_BS, 
                shuffle=True, 
                num_workers=cfg.NUM_WORKERS, 
                pin_memory=True, 
                drop_last=True
            )
        cfg.SOLVER.ITERATIONS_PER_EPOCH = len(loader['train'])

    if cfg.DB.USE_TEST:
        loader['query'] = DataLoader(query_dataset, 
                                    batch_size=cfg.INPUT.TEST_BS, 
                                    shuffle=False, 
                                    num_workers=cfg.NUM_WORKERS, 
                                    pin_memory=True, 
                                    drop_last=False)
        loader['gallery'] = DataLoader(gallery_dataset, 
                                    batch_size=cfg.INPUT.TEST_BS, 
                                    shuffle=False, 
                                    num_workers=cfg.NUM_WORKERS, 
                                    pin_memory=True, 
                                    drop_last=False)

    return loader