import os
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from engine.base_engine import BaseEngine, data_prefetcher
from tools.oracle_utils import gen_oracle_map
from tools.utils import multi_pose_decode, multi_pose_post_process
from tools.deepfashiontools.cocoeval import COCOeval as Clothingeval
from pycocotools.cocoeval import COCOeval as Personeval
import json
import numpy as np
import logging
logger = logging.getLogger("logger")
# recover = T.Compose([T.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225], std = [1/0.229,1/0.224,1/0.225])])

class CenterKPEngine(BaseEngine):
    def __init__(self, cfg, solvers, loader, show, manager):
        super(CenterKPEngine, self).__init__(cfg, solvers, loader, show, manager)

    def _train_epoch_start(self): 
        self.epoch += 1
        if self.cfg.SOLVER.MODEL_FREEZE_PEROID > 0:
            if self.epoch == 1:
                for n, p in self.core.named_parameters():
                    if n.find('hm') == -1 and n.find('wh') == -1 and n.find('reg') == -1:
                        p.requires_grad_(False)
                        logger.info("{:<60} Module Freezed".format(n))

            elif self.epoch == self.cfg.SOLVER.MODEL_FREEZE_PEROID + 1:
                for n, p in self.core.named_parameters():
                    if n.find('hm') == -1 and n.find('wh') == -1 and n.find('reg') == -1:
                        p.requires_grad_(True)
                        logger.info("{:<60} Module Thawed".format(n))

        logger.info("Epoch {} start".format(self.epoch))

        self.core.train() 

    def _train_once(self):
        for batch in tqdm(self.tdata, desc="Epoch[{}/{}]".format(self.epoch, self.max_epoch)):
            self._train_iter_start()
            for key in batch:
                batch[key] = batch[key].cuda()
            images = batch['inp']
            feats = self.core(images) 
            self.total_loss, self.each_loss = self.manager.loss_func(feats, batch)
            self.total_loss.backward()

            self._train_iter_end()

            self.total_loss = self.tensor_to_scalar(self.total_loss)
            self.each_loss = self.tensor_to_scalar(self.each_loss)
       

    def _evaluate(self, eval=False):
        logger.info("Epoch {} evaluation start".format(self.epoch))
        results = {}
        self._eval_epoch_start()
        with torch.no_grad():
            for batch in tqdm(self.vdata, desc="Validation"):
                for key in batch.keys():
                    batch[key] = batch[key].cuda()
                if self.cfg.ORACLE:
                    feat = {}
                    feat['hm']  = batch['hm']
                    feat['wh']  = torch.from_numpy(gen_oracle_map(batch['wh'].detach().cpu().numpy(), 
                                                                  batch['ind'].detach().cpu().numpy(), 
                                                                  batch['inp'].shape[3] // 8, batch['inp'].shape[2] // 8)).cuda()
                    feat['reg'] = torch.from_numpy(gen_oracle_map(batch['reg'].detach().cpu().numpy(), 
                                                                  batch['ind'].detach().cpu().numpy(), 
                                                                  batch['inp'].shape[3] // 8, batch['inp'].shape[2] // 8)).cuda()
                    feat['hm_hp'] = batch['hm_hp']
                    feat['hps'] = torch.from_numpy(gen_oracle_map(batch['hps'].detach().cpu().numpy(), 
                                                                  batch['ind'].detach().cpu().numpy(), 
                                                                  batch['inp'].shape[3] // 8, batch['inp'].shape[2] // 8)).cuda()
                    feat['hp_reg'] = torch.from_numpy(gen_oracle_map(batch['hp_reg'].detach().cpu().numpy(), 
                                                                     batch['hp_ind'].detach().cpu().numpy(), 
                                                                     batch['inp'].shape[3] // 8, batch['inp'].shape[2] // 8)).cuda()

                else:               
                    feat = self.core(batch['inp'])[-1]
                    feat['hm'].sigmoid_()
                    feat['hm_hp'].sigmoid_()

                dets = multi_pose_decode(feat['hm'], feat['wh'], feat['hps'], 
                                         reg=feat['reg'], hm_hp=feat['hm_hp'], hp_offset=feat['hp_reg'], K=100)
                dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])                    
                dets_out = multi_pose_post_process(dets.copy(), batch['c'].cpu().numpy(), batch['s'].cpu().numpy(),
                                                   feat['hm'].shape[2], feat['hm'].shape[3], feat['hm'].shape[1])
                results[batch['img_id'][0]] = dets_out[0]

        if self.cfg.DB.NUM_KEYPOINTS == 17:
            cce, cce_kp = coco_eval(Personeval, self.vdata.dataset.coco, results, self.cfg.OUTPUT_DIR)  
        elif self.cfg.DB.NUM_KEYPOINTS == 294:
            cce, cce_kp = coco_eval(Clothingeval, self.vdata.dataset.coco, results, self.cfg.OUTPUT_DIR)  

        logger.info('KP => Average Precision  (AP) @[ IoU=0.50:0.95 ] = {:.3f}'.format(cce_kp.stats[0]))
        logger.info('OD => Average Precision  (AP) @[ IoU=0.50:0.95 ] = {:.3f}'.format(cce.stats[0]))
        
        if not eval:
            self.accu = cce_kp.stats[0]
            self._eval_epoch_end()

    def Evaluate(self):
        self._evaluate(eval=True)

def _to_float(x):
    return float("{:.2f}".format(x))

def convert_eval_format(all_bboxes, valid_ids):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
        for cls_ind in all_bboxes[image_id]:
            for bbox in all_bboxes[image_id][cls_ind]:
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                score = bbox[4]
                bbox_out  = list(map(_to_float, bbox[0:4]))
                category_id = valid_ids[cls_ind - 1]
                keypoints = np.concatenate([np.array(bbox[5:], dtype=np.float32).reshape(-1, 2), 
                                            np.ones((len(bbox[5:])//2, 1), dtype=np.float32)], axis=1).reshape(3*(len(bbox[5:])//2)).tolist()
                keypoints  = list(map(_to_float, keypoints))

                detection = {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": bbox_out,
                    "score": float("{:.2f}".format(score)),
                    "keypoints": keypoints
                }
                detections.append(detection)
    return detections

def coco_eval(eval_func, coco, results, save_dir):
    json.dump(convert_eval_format(results, coco.getCatIds()), open('{}/results.json'.format(save_dir), 'w'))
    coco_dets = coco.loadRes('{}/results.json'.format(save_dir))
    coco_kp_eval = eval_func(coco, coco_dets, "keypoints")
    coco_kp_eval.evaluate()
    coco_kp_eval.accumulate()
    coco_kp_eval.summarize()
    coco_eval = eval_func(coco, coco_dets, "bbox")    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval, coco_kp_eval
