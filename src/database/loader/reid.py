from src.database.loader import *
from src.base_data import BaseData
from src.database.sampler.sampler import IdBasedSampler, IdBasedDistributedSampler

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

    data_names = cfg.DB.DATA.split(" ")
    if len(data_names) == 1:
        if use_train:
            train_data_names = data_names
        if use_test:
            test_data_name = data_names
    else:
        if use_train:
            train_data_names = data_names[:-1]
        if use_test:
            test_data_name = data_names[-1]

    loader = {}
    if use_train:         
        indice = []
        offset = 0
        for name in train_data_names:
            _data = DataFactory.produce(cfg, branch=name, use_test=False)
            for path, pid, cam in _data.train['indice']:
                indice.append((path, pid+offset, cam))
            offset += _data.train['n_samples']

        data = BaseData()
        data.train['indice'] = indice
        data.train['n_samples'] = offset
        cfg.REID.NUM_PERSON = offset
        train_trans = TransformFactory.produce(cfg, train_transformation)
        train_dataset = DataFormatFactory.produce(
            cfg, 
            data=data.train, 
            transform=train_trans, 
            return_indice=return_indice
        )
        if use_sampler:
            if cfg.DISTRIBUTED:
                sampler = IdBasedDistributedSampler(data.train['indice'], batch_size=train_batch_size, num_instances=num_people_per_batch)
            else:
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
    if use_test:
        data = DataFactory.produce(cfg, branch=test_data_name, use_train=False)      
        val_trans = TransformFactory.produce(cfg, test_transformation)      
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