from database.data import *
import os
import os.path as osp
import pycocotools.coco as coco
import json

class coco_data(BaseData):
    def __init__(self, cfg):
        self.dataset_dir = osp.join(cfg.DB.PATH, cfg.DB.DATA)
        self.category = cfg.COCO.TARGET
        self.train_dir   = osp.join(self.dataset_dir, "train2017")
        self.train_anno  = osp.join(self.dataset_dir, self.category, "instances_train2017.json")
        self.val_dir     = osp.join(self.dataset_dir, "val2017")
        self.val_anno    = osp.join(self.dataset_dir, self.category, "instances_val2017.json")
        self._check_before_run()
         
        if cfg.DB.USE_TRAIN:
            train_coco, train_images, train_num_samples = self._process_dir(self.train_anno, self.train_dir, split='train')
            self.handle['train'] = train_coco
            self.index_map['train'] = train_images
            self.n_samples['train'] = train_num_samples
            logger.info("=> COCO TRAIN is loaded")
            logger.info("  Dataset statistics:")
            logger.info("  -------------------")
            logger.info("  subset   | # images")
            logger.info("  -------------------")
            logger.info("  train    | {:8d}".format(train_num_samples))
            logger.info("  -------------------")
        if cfg.DB.USE_TEST:
            val_coco, val_images, val_num_samples = self._process_dir(self.val_anno, self.val_dir, split='val')
            self.handle['val'] = val_coco        
            self.index_map['val'] = val_images
            self.n_samples['val'] = val_num_samples
            logger.info("=> COCO VAL is loaded")
            logger.info("  Dataset statistics:")
            logger.info("  -------------------")
            logger.info("  subset   | # images")
            logger.info("  -------------------")
            logger.info("  val      | {:8d}".format(val_num_samples))
            logger.info("  -------------------")
        
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.train_anno):
            orig_json = osp.join(self.dataset_dir, 'original', "instances_train2017.json")
            if not osp.exists(orig_json):
                raise RuntimeError("'{}' is not available to make target of coco".format(orig_json))
            self._make_target_coco(orig_json, self.train_anno, self.category)
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))        
        if not osp.exists(self.val_anno):
            orig_json = osp.join(self.dataset_dir, 'original', "instances_val2017.json")
            if not osp.exists(orig_json):
                raise RuntimeError("'{}' is not available to make target of coco".format(orig_json))
            self._make_target_coco(orig_json, self.val_anno, self.category)
    
    def _process_dir(self, anno_path, img_path, split='train'):
        data_handle = coco.COCO(anno_path)
        image_ids = data_handle.getImgIds()
        images = []
        for img_id in image_ids:
            idxs = data_handle.getAnnIds(imgIds=[img_id])
            if split == 'train' and len(idxs) > 0:
                fname = data_handle.loadImgs(ids=[img_id])[0]['file_name']
                fname = osp.join(img_path, fname)
                images.append((img_id, fname))
            else:
                fname = data_handle.loadImgs(ids=[img_id])[0]['file_name']
                fname = osp.join(img_path, fname)
                images.append((img_id, fname))
     
        num_samples = len(images)
        
        return data_handle, images, num_samples
    
    def _make_target_coco(self, src, dst, category):
        logger.info("Making target of coco of {} ...".format(dst))
        if not osp.exists(osp.dirname(dst)):
            os.mkdir(osp.dirname(dst))
        anno_path = src

        f = open(anno_path, 'r')
        coco_json = json.loads(f.readline())
        coco_trt = {}
        coco_trt['info'] = coco_json['info']
        coco_trt['licenses'] = coco_json['licenses']
        coco_trt['categories'] = [coco_json['categories'][0]]

        del coco_json

        coco_data = coco.COCO(anno_path)
        cat_ids = coco_data.getCatIds(catNms=[category])
        trt_img_ids = coco_data.getImgIds(catIds=cat_ids)
        annids = coco_data.getAnnIds(imgIds=trt_img_ids)
        anns = coco_data.loadAnns(ids=annids)
        trt_anns = []
        for ann in anns:
            if ann['category_id'] == cat_ids[0]:
                trt_anns.append(ann)
        trt_imgs = coco_data.loadImgs(ids=trt_img_ids)
        
        coco_trt['images'] = trt_imgs
        coco_trt['annotations'] = trt_anns

        json.dump(coco_trt, open(dst, 'w'))