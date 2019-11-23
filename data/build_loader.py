from torch.utils import data
from data.data import get_img_data, init_vid_dataset
from data.build_data import *
from data.build_transform import build_transform
from data.sampler import IdBasedSampler, BlancedPARSampler
from torchvision.datasets.cifar import CIFAR10
import logging

def build_imagenet_loader(cfg):

    dataset = get_img_data(cfg)

    train_trans = build_transform(cfg)
    val_trans = build_transform(cfg, is_train=False)

    train_dataset = build_image_dataset(dataset.train, train_trans)
    val_dataset = build_image_dataset(dataset.val, val_trans)

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
    val_trans = build_transform(cfg, is_train=False)

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

def build_reid_loader(cfg, return_indice=False, use_sampler=True):
    
    if "merge" in cfg.DATASET.NAME:
        dataset_names = cfg.DATASET.NAME.split(" ")[1:]

        temp_datasets = []
        for name in dataset_names:
            cfg.DATASET.NAME = name
            dataset = get_img_data(cfg)
            temp_datasets.append(dataset)
            cfg.MODEL.NUM_CLASSES += dataset.num_train_pids

        dataset = []

        offset = 0
        for i in range(len(temp_datasets)):   
            for path, pid, cam in temp_datasets[i].train: 
                dataset.append([path, pid+offset, cam])
            offset += temp_datasets[i].num_train_pids
        
        train_trans = build_transform(cfg)
        val_trans = build_transform(cfg, is_train=False)

        train_dataset = build_reid_dataset(dataset, train_trans)

        cfg.DATASET.NAME = cfg.REID.TRT
        trt_dataset = get_img_data(cfg)
        query_dataset = build_reid_dataset(trt_dataset.query, val_trans)
        gallery_dataset = build_reid_dataset(trt_dataset.gallery, val_trans)
    
    else:
        dataset = get_img_data(cfg)
        train_trans = build_transform(cfg)
        val_trans = build_transform(cfg, is_train=False)

        if cfg.DATASET.ATTENTION_MAPS != "":
            train_dataset = build_reid_atmap_dataset(dataset.train, cfg)
        else:
            train_dataset = build_reid_dataset(dataset.train, train_trans, return_indice=return_indice)

        if cfg.DATASET.TEST != "":
            cfg.DATASET.NAME = cfg.DATASET.TEST
            dataset = get_img_data(cfg)

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
        if use_sampler:
            if isinstance(dataset, list):
                sampler = IdBasedSampler(dataset, batch_size=cfg.INPUT.SIZE_TRAIN, num_instances=cfg.REID.SIZE_PERSON)
            else:
                sampler = IdBasedSampler(dataset.train, batch_size=cfg.INPUT.SIZE_TRAIN, num_instances=cfg.REID.SIZE_PERSON)

            t_loader = data.DataLoader(
                train_dataset, 
                batch_size=cfg.INPUT.SIZE_TRAIN, 
                sampler=sampler, 
                num_workers=num_workers, 
                pin_memory=True, 
                drop_last=True
            )
        else:
            t_loader = data.DataLoader(
                train_dataset, 
                batch_size=cfg.INPUT.SIZE_TRAIN, 
                shuffle=True, 
                num_workers=num_workers, 
                pin_memory=True, 
                drop_last=True
            )

    q_loader = data.DataLoader(
        query_dataset, batch_size=cfg.INPUT.SIZE_TEST, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    g_loader = data.DataLoader(
        gallery_dataset, batch_size=cfg.INPUT.SIZE_TEST, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    return t_loader, q_loader, g_loader
    
def build_par_loader(cfg):

    dataset = get_img_data(cfg)

    train_trans = build_transform(cfg)
    val_trans = build_transform(cfg, is_train=False)

    train_dataset = build_par_dataset(dataset.train, train_trans)
    val_dataset = build_par_dataset(dataset.val, val_trans)

    num_workers = cfg.DATALOADER.NUM_WORKERS

    par_sampler = BlancedPARSampler(dataset)
    t_loader = data.DataLoader(
        train_dataset, batch_size=cfg.INPUT.SIZE_TRAIN, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=par_sampler
    )

    v_loader = data.DataLoader(
        val_dataset, batch_size=cfg.INPUT.SIZE_TEST, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return t_loader, v_loader

def build_update_reid_loader(cfg, new_labels):
    dataset = get_img_data(cfg)

    train_trans = build_transform(cfg)
    val_trans = build_transform(cfg, is_train=False)

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
        
    dataset = get_img_data(cfg)
    train_trans = build_transform(cfg)
    val_trans = build_transform(cfg, is_train=False)

    train_dataset = build_reid_dataset(dataset.train, train_trans)

    query_dataset = build_reid_dataset(dataset.query, val_trans)
    gallery_dataset = build_reid_dataset(dataset.gallery, val_trans)    

    num_workers = cfg.DATALOADER.NUM_WORKERS

    t_loader = data.DataLoader(
        train_dataset, 
        batch_size=cfg.INPUT.SIZE_TRAIN, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False
    )
    q_loader = data.DataLoader(
        query_dataset, batch_size=cfg.INPUT.SIZE_TEST, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    g_loader = data.DataLoader(
        gallery_dataset, batch_size=cfg.INPUT.SIZE_TEST, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    return t_loader, q_loader, g_loader

def build_coco_person_loader(cfg):
    data = get_img_data(cfg)

    train_dataset = build_COCO_Person_dataset(data.train_coco, data.train_images, src=data.train_dir, split='train')
    val_dataset = build_COCO_Person_dataset(data.val_coco, data.val_images, src=data.val_dir, split='val')

    num_workers = cfg.DATALOADER.NUM_WORKERS

    t_loader = data.DataLoader(
        train_dataset, 
        batch_size=cfg.INPUT.SIZE_TRAIN, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False
    )
    v_loader = data.DataLoader(
        val_dataset, batch_size=cfg.INPUT.SIZE_TEST, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    return t_loader, v_loader