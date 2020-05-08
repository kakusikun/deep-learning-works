from functools import partial
from src.database.loader import *
from src.base_data import BaseData
from tools.centerface_utils import centerface_facial_target, centerface_bbox_target
from tools.centernet_utils import centernet_keypoints_target, centernet_bbox_target
from tools.yolov3_utils import yolov3_JDE_targets
import re
from torch._six import container_abcs, string_classes, int_classes
import numpy as np
np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

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
    data_names = cfg.DB.DATA.split(" ")
    if len(data_names) == 1:
        if use_train:
            train_data_names = data_names
        if use_test:
            test_data_name = data_names[0]
    else:
        if use_train:
            train_data_names = data_names[:-1]
        if use_test:
            test_data_name = data_names[-1]

    loader = {}
    if target_format == 'centernet_kp':
        build_func = centernet_keypoints_target
    elif target_format == 'centerface':
        build_func = centerface_facial_target
    elif target_format == 'centernet':
        build_func = centernet_bbox_target
    elif target_format == 'centerface_bbox':
        build_func = centerface_bbox_target
    elif target_format == 'yolov3_jde':
        build_func = partial(yolov3_JDE_targets, anchors=cfg.YOLO.ANCHORS)
    else:
        build_func = None
    if use_train:
        handles = []
        indice = []
        pids = []
        offset = 0
        
        for idx, name in enumerate(train_data_names):
            _data = DataFactory.produce(cfg, branch=name, use_test=False)
            handles.append(_data.train['handle'])
            pids.append(_data.train['pid'])
            for img_id, img_path in _data.train['indice']:
                indice.append((img_id, img_path, idx, offset))
            offset += _data.train['num_person']

        data = BaseData()
        data.train['handle'] = handles
        data.train['indice'] = indice
        data.train['pid'] = pids
        data.train['strides'] = cfg.MODEL.STRIDES
        data.train['num_classes'] = cfg.DB.NUM_CLASSES
        data.train['num_keypoints'] = cfg.DB.NUM_KEYPOINTS
        data.train['num_person'] = offset
        cfg.REID.NUM_PERSON = offset
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
            drop_last=True,
            collate_fn=default_collate
        )
    if use_test:
        data = DataFactory.produce(cfg, branch=test_data_name, use_train=False) 
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
            drop_last=False,
            collate_fn=default_collate,
        ) 
    return loader


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        output = {}
        for key in elem:
            if 'bboxes' in key:
                output[key] = [torch.from_numpy(np.vstack(d[key])) for d in batch]
            else:
                output[key] = default_collate([d[key] for d in batch])
        return output
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
    

if __name__ == "__main__":
    from src.factory.config_factory import cfg
    from src.factory.loader_factory import LoaderFactory

    cfg.NUM_WORKERS = 1
    cfg.DB.PATH = "/media/allen/mass/DB"
    cfg.DB.DATA = "prw coco"
    cfg.DB.DATA_FORMAT = "coco_reid"
    cfg.DB.TARGET_FORMAT = "yolov3_jde"
    cfg.DB.LOADER = "coco_reid"
    cfg.DB.USE_TRAIN = True
    cfg.DB.USE_TEST = False
    cfg.INPUT.SIZE = (576, 320)
    cfg.INPUT.TEST_BS = 4
    cfg.MODEL.STRIDES = [8, 16, 32]
    cfg.DB.TRAIN_TRANSFORM = "ResizeKeepAspectRatio Tensorize"
    cfg.DB.TEST_TRANSFORM = "Resize Tensorize"
    cfg.REID.MSMT_ALL = False
    cfg.YOLO.ANCHORS = [6,16, 8,23, 11,32, 16,45,   21,64, 30,90, 43,128, 60,180,   85,255, 120,360, 170,420, 340, 320]

    loader = LoaderFactory.produce(cfg)

    batch = next(iter(loader['train']))
    # print(batch)