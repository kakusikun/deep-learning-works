from src.database.loader import *
import os.path as osp

def build_classification_loader(
    cfg, 
    use_train=False, 
    use_test=False,
    train_transformation="", 
    test_transformation="",
    train_batch_size=-1, 
    test_batch_size=-1,
    num_workers=-1, 
    **kwargs):
    data = DataFactory.produce(cfg)
    loader = {}
    if use_train:
        train_trans = TransformFactory.produce(cfg, train_transformation)
        train_dataset = DataFormatFactory.produce(cfg, data=data.train, transform=train_trans)
        if cfg.DISTRIBUTED:
            sampler = distributed.DistributedSampler(train_dataset)
        else:
            sampler = None
        loader['train'] = DataLoader(
            train_dataset, 
            batch_size=train_batch_size, 
            shuffle=True if sampler is None else False, 
            num_workers=num_workers, 
            pin_memory=False,
            drop_last=True, 
            sampler=sampler,
        )
    if use_test:
        val_trans = TransformFactory.produce(cfg, test_transformation)
        val_dataset = DataFormatFactory.produce(cfg, data=data.val, transform=val_trans)
        loader['val'] = DataLoader(
            val_dataset, 
            batch_size=test_batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=False,
            drop_last=False
        )

    return loader