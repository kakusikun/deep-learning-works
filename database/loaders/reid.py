from database.loaders import *
from database.data_factory import get_data
from database.dataset_factory import get_dataset
from database.transform_factory import get_transform
from database.sampler.sampler import IdBasedSampler

def build_reid_loader(cfg, return_indice=False, use_sampler=True):
    loader = {}
    val_trans = get_transform(cfg, cfg.TEST_TRANSFORM)
    train_trans = get_transform(cfg, cfg.TRAIN_TRANSFORM)

    if "merge" in cfg.DB.DATA:
        data_names = cfg.DB.DATA.split(" ")[1:]
        if cfg.DB.USE_TRAIN:   
            _data = [] 
            offset = 0
            for name in data_names[:-1]:
                cfg.DB.DATA = name
                data = get_data(name)(cfg)
                cfg.DB.NUM_CLASSES += data.train['n_samples']
                for path, pid, cam in data.train['indice']: 
                    _data.append([path, pid+offset, cam])
                offset += data.train['n_samples']
            data.train['indice'] = _data
            data.train['n_samples'] = offset
            
            train_dataset = get_dataset(cfg.DB.DATASET)(data.train, train_trans)
            if use_sampler:
                sampler = IdBasedSampler(data.train['indice'], batch_size=cfg.INPUT.TRAIN_BS, num_instances=cfg.REID.SIZE_PERSON)
        if cfg.DB.USE_TEST:
            data = get_data(data_names[-1])(cfg)            
            query_dataset = get_dataset(cfg.DB.DATASET)(data.query, val_trans)
            gallery_dataset = get_dataset(cfg.DB.DATASET)(data.gallery, val_trans)
    else:
        data = get_data(cfg.DB.DATA)(cfg)
        if cfg.DB.USE_TRAIN:
            train_dataset = get_dataset(cfg.DB.DATASET)(data.train, train_trans)
            cfg.DB.NUM_CLASSES = data.n_samples['train']
            if use_sampler:
                sampler = IdBasedSampler(data.train['indice'], batch_size=cfg.INPUT.TRAIN_BS, num_instances=cfg.REID.SIZE_PERSON)
        if cfg.DB.USE_TEST: 
            query_dataset = get_dataset(cfg.DB.DATASET)(data.query, val_trans)
            gallery_dataset = get_dataset(cfg.DB.DATASET)(data.gallery, val_trans)

    if cfg.DB.USE_TRAIN: 
        if use_sampler:           
            loader['train'] = Data.DataLoader(
                train_dataset, 
                batch_size=cfg.INPUT.TRAIN_BS, 
                sampler=sampler, 
                num_workers=cfg.NUM_WORKERS, 
                pin_memory=True, 
                drop_last=True
            )
        else:
            loader['train'] = Data.DataLoader(
                train_dataset, 
                batch_size=cfg.INPUT.TRAIN_BS, 
                shuffle=True, 
                num_workers=cfg.NUM_WORKERS, 
                pin_memory=True, 
                drop_last=True
            )
        cfg.SOLVER.ITERATIONS_PER_EPOCH = len(loader['train'])

    if cfg.DB.USE_TEST:
        loader['query'] = Data.DataLoader(query_dataset, 
                                    batch_size=cfg.INPUT.TEST_BS, 
                                    shuffle=False, 
                                    num_workers=cfg.NUM_WORKERS, 
                                    pin_memory=True, 
                                    drop_last=False)
        loader['gallery'] = Data.DataLoader(gallery_dataset, 
                                    batch_size=cfg.INPUT.TEST_BS, 
                                    shuffle=False, 
                                    num_workers=cfg.NUM_WORKERS, 
                                    pin_memory=True, 
                                    drop_last=False)

    return loader