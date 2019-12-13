from torch.utils import data
from data.data import get_img_data, init_vid_dataset
from data.build_data import *
from data.build_transform import build_transform
from data.sampler import IdBasedSampler, BlancedPARSampler
from torchvision.datasets.cifar import CIFAR10
import logging
import sys

def build_imagenet_loader(cfg):

    dataset = get_img_data(cfg)

    train_trans = build_transform(cfg)
    val_trans = build_transform(cfg, is_train=False)

    train_dataset = build_image_dataset(dataset.train, train_trans)
    val_dataset = build_image_dataset(dataset.val, val_trans)

    num_workers = cfg.NUM_WORKERS

    t_loader = Data.DataLoader(
        train_dataset, batch_size=cfg.INPUT.TRAIN_BS, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    v_loader = Data.DataLoader(
        val_dataset, batch_size=cfg.INPUT.TEST_BS, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return t_loader, v_loader

def build_cifar10_loader(cfg):


    train_trans = build_transform(cfg)
    val_trans = build_transform(cfg, is_train=False)

    train_dataset = CIFAR10(root=cfg.DB.PATH, train=True, transform=train_trans, download=True)
    val_dataset = CIFAR10(root=cfg.DB.PATH, train=False, transform=val_trans, download=True)

    num_workers = cfg.NUM_WORKERS

    t_loader = Data.DataLoader(
        train_dataset, batch_size=cfg.INPUT.TRAIN_BS, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    v_loader = Data.DataLoader(
        val_dataset, batch_size=cfg.INPUT.TEST_BS, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return t_loader, v_loader


    
def build_par_loader(cfg):

    dataset = get_img_data(cfg)

    train_trans = build_transform(cfg)
    val_trans = build_transform(cfg, is_train=False)

    train_dataset = build_par_dataset(dataset.train, train_trans)
    val_dataset = build_par_dataset(dataset.val, val_trans)

    num_workers = cfg.NUM_WORKERS

    par_sampler = BlancedPARSampler(dataset)
    t_loader = Data.DataLoader(
        train_dataset, batch_size=cfg.INPUT.TRAIN_BS, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=par_sampler
    )

    v_loader = Data.DataLoader(
        val_dataset, batch_size=cfg.INPUT.TEST_BS, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return t_loader, v_loader

def build_update_reid_loader(cfg, new_labels):
    dataset = get_img_data(cfg)

    train_trans = build_transform(cfg)
    val_trans = build_transform(cfg, is_train=False)

    train_dataset = build_update_reid_dataset(new_labels, dataset.train, train_trans)
    query_dataset = build_reid_dataset(dataset.query, val_trans)
    gallery_dataset = build_reid_dataset(dataset.gallery, val_trans)    

    num_workers = cfg.NUM_WORKERS

    sampler = IdBasedSampler(train_dataset, batch_size=cfg.INPUT.TRAIN_BS, num_instances=cfg.REID.SIZE_PERSON)

    t_loader = Data.DataLoader(
        train_dataset, 
        batch_size=cfg.INPUT.TRAIN_BS, 
        sampler=sampler, 
        num_workers=num_workers, 
        pin_memory=True
    )
    q_loader = Data.DataLoader(
        query_dataset, batch_size=cfg.INPUT.TEST_BS, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    g_loader = Data.DataLoader(
        gallery_dataset, batch_size=cfg.INPUT.TEST_BS, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )

    return t_loader, q_loader, g_loader

def build_plain_reid_loader(cfg):
        
    dataset = get_img_data(cfg)
    train_trans = build_transform(cfg)
    val_trans = build_transform(cfg, is_train=False)

    train_dataset = build_reid_dataset(dataset.train, train_trans)

    query_dataset = build_reid_dataset(dataset.query, val_trans)
    gallery_dataset = build_reid_dataset(dataset.gallery, val_trans)    

    num_workers = cfg.NUM_WORKERS

    t_loader = Data.DataLoader(
        train_dataset, 
        batch_size=cfg.INPUT.TRAIN_BS, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False
    )
    q_loader = Data.DataLoader(
        query_dataset, batch_size=cfg.INPUT.TEST_BS, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    g_loader = Data.DataLoader(
        gallery_dataset, batch_size=cfg.INPUT.TEST_BS, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    return t_loader, q_loader, g_loader

def build_coco_loader(cfg):
    dataset = get_img_data(cfg)

    train_dataset = build_COCO_dataset(dataset.train_coco, dataset.train_images, split='train')
    val_dataset = build_COCO_dataset(dataset.val_coco, dataset.val_images, split='val')

    num_workers = cfg.NUM_WORKERS

    t_loader = Data.DataLoader(
        train_dataset, 
        batch_size=cfg.INPUT.TRAIN_BS, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True
    )
    v_loader = Data.DataLoader(
        val_dataset, batch_size=cfg.INPUT.TEST_BS, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    return t_loader, v_loader


