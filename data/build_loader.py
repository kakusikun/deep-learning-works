from torch.utils import data
from data.data_manager import init_img_dataset, init_vid_dataset
from data.build_data import build_image_dataset, build_reid_dataset, build_reid_atmap_dataset, build_par_dataset, build_update_reid_dataset
from data.build_transform import build_transform
from data.sampler import IdBasedSampler
from torchvision.datasets.cifar import CIFAR10
import logging

def build_imagenet_loader(cfg):

    dataset = init_img_dataset(cfg)

    train_trans = build_transform(cfg)
    val_trans = build_transform(cfg, isTrain=False)

    train_dataset = build_image_dataset(dataset.train, train_trans, dataset.train_lmdb)
    val_dataset = build_image_dataset(dataset.val, val_trans, dataset.val_lmdb)

    num_workers = cfg.DATALOADER.NUM_WORKERS

    t_loader = data.DataLoader(
        train_dataset, batch_size=cfg.INPUT.SIZE_TRAIN, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    v_loader = data.DataLoader(
        val_dataset, batch_size=cfg.INPUT.SIZE_TEST, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return t_loader, v_loader

def build_cifar10_loader(cfg):


    train_trans = build_transform(cfg)
    val_trans = build_transform(cfg, isTrain=False)

    train_dataset = CIFAR10(root=cfg.DATASET.TRAIN_PATH, train=True, transform=train_trans, download=True)
    val_dataset = CIFAR10(root=cfg.DATASET.TRAIN_PATH, train=False, transform=val_trans, download=True)

    num_workers = cfg.DATALOADER.NUM_WORKERS

    t_loader = data.DataLoader(
        train_dataset, batch_size=cfg.INPUT.SIZE_TRAIN, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    v_loader = data.DataLoader(
        val_dataset, batch_size=cfg.INPUT.SIZE_TEST, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return t_loader, v_loader

def build_reid_loader(cfg):
    
    if cfg.DATASET.NAME == "total":
        cfg.DATASET.NAME = 'cuhk03'
        cuhk_dataset = init_img_dataset(cfg)

        cfg.DATASET.NAME = 'market1501'
        market_dataset = init_img_dataset(cfg)

        cfg.DATASET.NAME = 'dukemtmcreid'
        duke_dataset = init_img_dataset(cfg)

        _dataset = []
        _dataset.extend(cuhk_dataset.train)
        _dataset.extend(market_dataset.train)
        _dataset.extend(duke_dataset.train)

        dataset = []

        for i, (a, b, c) in enumerate(_dataset):   
            
            if 'market' in a:
                offset = cuhk_dataset.num_train_pids
                b += offset
            if 'duke' in a:
                offset = cuhk_dataset.num_train_pids + market_dataset.num_train_pids
                b += offset
            dataset.append([a, b, c])
        
        train_trans = build_transform(cfg)
        val_trans = build_transform(cfg, isTrain=False)

        train_dataset = build_reid_dataset(dataset, train_trans)
        query_dataset = build_reid_dataset(market_dataset.query, val_trans)
        gallery_dataset = build_reid_dataset(market_dataset.gallery, val_trans)
    
    elif cfg.DATASET.NAME == "general":     
        cfg.DATASET.NAME = 'market1501'
        market_dataset = init_img_dataset(cfg)

        cfg.DATASET.NAME = 'msmt17'
        msmt_dataset = init_img_dataset(cfg)
        
        train_trans = build_transform(cfg)
        val_trans = build_transform(cfg, isTrain=False)

        train_dataset = build_reid_dataset(msmt_dataset.train, train_trans)
        query_dataset = build_reid_dataset(market_dataset.query, val_trans)
        gallery_dataset = build_reid_dataset(market_dataset.gallery, val_trans)
    else:
        dataset = init_img_dataset(cfg)
        train_trans = build_transform(cfg)
        val_trans = build_transform(cfg, isTrain=False)

        if cfg.DATASET.ATTENTION_MAPS != "":
            train_dataset = build_reid_atmap_dataset(dataset.train, cfg)
        else:
            train_dataset = build_reid_dataset(dataset.train, train_trans)

        if cfg.DATASET.TEST != "":
            cfg.DATASET.NAME = cfg.DATASET.TEST
            dataset = init_img_dataset(cfg)

        query_dataset = build_reid_dataset(dataset.query, val_trans)
        gallery_dataset = build_reid_dataset(dataset.gallery, val_trans)    

    num_workers = cfg.DATALOADER.NUM_WORKERS


    if cfg.EVALUATE != "":
        t_loader = data.DataLoader(
            train_dataset, 
            batch_size=cfg.INPUT.SIZE_TRAIN,
            num_workers=num_workers, 
            pin_memory=True
        )
    else:
        sampler = IdBasedSampler(dataset.train, batch_size=cfg.INPUT.SIZE_TRAIN, num_instances=cfg.REID.SIZE_PERSON)

        t_loader = data.DataLoader(
            train_dataset, 
            batch_size=cfg.INPUT.SIZE_TRAIN, 
            sampler=sampler, 
            num_workers=num_workers, 
            pin_memory=True
        )

    q_loader = data.DataLoader(
        query_dataset, batch_size=cfg.INPUT.SIZE_TEST, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    g_loader = data.DataLoader(
        gallery_dataset, batch_size=cfg.INPUT.SIZE_TEST, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    return t_loader, q_loader, g_loader

def build_par_loader(cfg):

    dataset = init_img_dataset(cfg)

    train_trans = build_transform(cfg)
    val_trans = build_transform(cfg, isTrain=False)

    train_dataset = build_par_dataset(dataset.train, train_trans)
    val_dataset = build_par_dataset(dataset.val, val_trans)

    num_workers = cfg.DATALOADER.NUM_WORKERS

    t_loader = data.DataLoader(
        train_dataset, batch_size=cfg.INPUT.SIZE_TRAIN, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    v_loader = data.DataLoader(
        val_dataset, batch_size=cfg.INPUT.SIZE_TEST, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return t_loader, v_loader

def build_update_reid_loader(cfg, new_labels):
    dataset = init_img_dataset(cfg)

    train_trans = build_transform(cfg)
    val_trans = build_transform(cfg, isTrain=False)

    train_dataset = build_update_reid_dataset(new_labels, dataset.train, train_trans)
    query_dataset = build_reid_dataset(dataset.query, val_trans)
    gallery_dataset = build_reid_dataset(dataset.gallery, val_trans)    

    num_workers = cfg.DATALOADER.NUM_WORKERS

    sampler = IdBasedSampler(train_dataset, batch_size=cfg.INPUT.SIZE_TRAIN, num_instances=cfg.REID.SIZE_PERSON)

    t_loader = data.DataLoader(
        train_dataset, 
        batch_size=cfg.INPUT.SIZE_TRAIN, 
        sampler=sampler, 
        num_workers=num_workers, 
        pin_memory=True
    )
    q_loader = data.DataLoader(
        query_dataset, batch_size=cfg.INPUT.SIZE_TEST, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    g_loader = data.DataLoader(
        gallery_dataset, batch_size=cfg.INPUT.SIZE_TEST, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )

    return t_loader, q_loader, g_loader


def build_plain_reid_loader(cfg):
        
    dataset = init_img_dataset(cfg)
    train_trans = build_transform(cfg)
    val_trans = build_transform(cfg, isTrain=False)

    train_dataset = build_reid_dataset(dataset.train, train_trans)

    query_dataset = build_reid_dataset(dataset.query, val_trans)
    gallery_dataset = build_reid_dataset(dataset.gallery, val_trans)    

    num_workers = cfg.DATALOADER.NUM_WORKERS

    t_loader = data.DataLoader(
        train_dataset, 
        batch_size=cfg.INPUT.SIZE_TRAIN, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    q_loader = data.DataLoader(
        query_dataset, batch_size=cfg.INPUT.SIZE_TEST, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    g_loader = data.DataLoader(
        gallery_dataset, batch_size=cfg.INPUT.SIZE_TEST, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    return t_loader, q_loader, g_loader
