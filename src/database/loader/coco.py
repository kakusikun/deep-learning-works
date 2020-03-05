from src.database.loader import *
from tools.centerface_utils import centerface_facial_target, centerface_bbox_target
from tools.centernet_utils import centernet_keypoints_target, centernet_bbox_target

def build_coco_loader(
        cfg, 
        target_format="",
        use_train=False, 
        use_test=False,
        train_transformation="", 
        test_transformation="",
        train_batch_size=-1, 
        test_batch_size=-1,
        num_workers=-1, 
        **kwargs):

    data = DataFactory.produce(cfg)
    if target_format == 'centernet_kp':
        build_func = centernet_keypoints_target
    elif target_format == 'centerface':
        build_func = centerface_facial_target
    elif target_format == 'centernet':
        build_func = centernet_bbox_target
    elif target_format == 'centerface_bbox':
        build_func = centerface_bbox_target   
    else:
        build_func = None
    loader = {}
    if use_train:
        train_trans = TransformFactory.produce(cfg, train_transformation)
        train_dataset = DataFormatFactory.produce(
            cfg, 
            data=data.train, 
            transform=train_trans, 
            build_func=build_func
        )
        loader['train'] = DataLoader(
            train_dataset, 
            batch_size=train_batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=False,
            drop_last=True
        )
    if use_test:
        val_trans = TransformFactory.produce(cfg, test_transformation)
        val_dataset = DataFormatFactory.produce(
            cfg, 
            data=data.val, 
            transform=val_trans,
            build_func=build_func
        )
        loader['val'] = DataLoader(
            val_dataset, 
            batch_size=test_batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=False, 
            drop_last=False
        ) 
    return loader
