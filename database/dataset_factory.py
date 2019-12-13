from database.datasets.coco_kp import build_cocokp_dataset
from database.datasets.coco_ob import build_coco_dataset
from database.datasets.reid import build_reid_dataset
from database.datasets.imagenet import build_image_dataset

dataset_factory = {
    'coco_keypoint': build_cocokp_dataset,
    'coco_object': build_coco_dataset,
    'reid': build_reid_dataset,
    'imagenet': build_image_dataset,    
}

def get_names():
    return list(dataset_factory.keys())

def get_dataset(name):
    if name not in dataset_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, dataset_factory.keys()))   
    return dataset_factory[name]
