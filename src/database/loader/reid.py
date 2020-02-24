from src.database.loader import *
from src.database.sampler.sampler import IdBasedSampler

def build_reid_loader(
    cfg, 
    num_people_per_batch=-1,
    use_train=False, 
    use_test=False,
    train_transformation="", 
    test_transformation="",
    train_batch_size=-1, 
    test_batch_size=-1,
    num_workers=-1, 
    return_indice=False, 
    use_sampler=True,
    **kwargs):

    loader = {}
    data_names = cfg.DB.DATA.split(" ")[1:]
    if use_train: 
        if len(data_names) > 1:
            _data = [] 
            offset = 0
            for name in data_names[:-1]:
                data = DataFactory.produce(cfg, name=name)
                #TODO: move to config checking
                cfg.DB.NUM_CLASSES += data.train['n_samples']
                for path, pid, cam in data.train['indice']:
                    _data.append([path, pid+offset, cam])
                offset += data.train['n_samples']
            data.train['indice'] = _data
            data.train['n_samples'] = offset
        else:
            data = DataFactory.produce(cfg)

        train_trans = TransformFactory.produce(cfg, test_transformation)
        train_dataset = DataFormatFactory.produce(
            cfg, 
            data=data.train, 
            transform=train_trans, 
            return_indice=return_indice
        )
        if use_sampler:    
            sampler = IdBasedSampler(data.train['indice'], batch_size=train_batch_size, num_instances=num_people_per_batch)       
            loader['train'] = DataLoader(
                train_dataset, 
                batch_size=train_batch_size, 
                sampler=sampler, 
                num_workers=num_workers, 
                pin_memory=False, 
                drop_last=True
            )
        else:
            loader['train'] = DataLoader(
                train_dataset, 
                batch_size=train_batch_size, 
                shuffle=True, 
                num_workers=num_workers, 
                pin_memory=False, 
                drop_last=True
            )
        #TODO: move to config checking
        cfg.SOLVER.ITERATIONS_PER_EPOCH = len(loader['train'])

    if use_test:
        data = DataFactory.produce(cfg, name=data_names[-1])      
        val_trans = TransformFactory.produce(cfg, train_transformation)      
        query_dataset = DataFormatFactory.produce(cfg, data=data.query, transform=val_trans)
        gallery_dataset = DataFormatFactory.produce(cfg, data=data.gallery, transform=val_trans)

        loader['query'] = DataLoader(query_dataset, 
                                    batch_size=test_batch_size, 
                                    shuffle=False, 
                                    num_workers=num_workers, 
                                    pin_memory=False, 
                                    drop_last=False)
        loader['gallery'] = DataLoader(gallery_dataset, 
                                    batch_size=test_batch_size, 
                                    shuffle=False, 
                                    num_workers=num_workers, 
                                    pin_memory=False, 
                                    drop_last=False)

    return loader