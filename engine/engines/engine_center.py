import os
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from engine.engine import Engine, data_prefetcher
from tools.oracle_utils import gen_oracle_map
from tools.utils import ctdet_decode, ctdet_post_process
from pycocotools.cocoeval import COCOeval
import json
import numpy as np
import logging
logger = logging.getLogger("logger")
# recover = T.Compose([T.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225], std = [1/0.229,1/0.224,1/0.225])])

class CenterEngine(Engine):
    def __init__(self, cfg, opts, tdata, vdata, show, manager):
        super(CenterEngine, self).__init__(cfg, opts, tdata, vdata, None, None, show, manager)

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

    def _train_iter_start(self):
        self.iter += 1
        for opt in self.opts:
            opt.lr_adjust(self.total_loss, self.iter)
            opt.zero_grad()

    def _train_iter_end(self):  
        for opt in self.opts:
            opt.step()

        self.show.add_scalar('train/total_loss', self.total_loss, self.iter)              
        for i in range(len(self.each_loss)):
            self.show.add_scalar('train/loss/{}'.format(self.manager.loss_name[i]), self.each_loss[i], self.iter)
        self.show.add_scalar('train/accuracy', self.train_accu, self.iter)   
        for i in range(len(self.opts)):
            self.show.add_scalar('train/opt/{}/lr'.format(i), self.opts[i].monitor_lr, self.iter)

    def _train_once(self):
        prefetcher = data_prefetcher(self.tdata)
        for _ in tqdm(range(len(self.tdata)), desc="Epoch[{}/{}]".format(self.epoch, self.max_epoch)):
            self._train_iter_start()

            batch = prefetcher.next()
            if batch is None:
                break
            images = batch['inp']
            feats = self.core(images) 
            self.total_loss, self.each_loss = self.manager.loss_func(feats, batch)
            self.total_loss.backward()

            self._train_iter_end()

            self.total_loss = self.tensor_to_scalar(self.total_loss)
            self.each_loss = self.tensor_to_scalar(self.each_loss)
       

    def _evaluate(self, eval=False):
        logger.info("Epoch {} evaluation start".format(self.epoch))
        prefetcher = data_prefetcher(self.vdata)
        results = {}
        self._eval_epoch_start()
        with torch.no_grad():
            for _ in tqdm(range(len(self.vdata)), desc="Validation"):                
                batch = prefetcher.next()
                if batch is None:
                    break
                if self.cfg.ORACLE:
                    feat = {}
                    feat['hm']  = batch['hm']
                    feat['wh']  = torch.from_numpy(gen_oracle_map(batch['wh'].detach().cpu().numpy(), 
                                                                  batch['ind'].detach().cpu().numpy(), 
                                                                  batch['inp'].shape[3] // 8, batch['inp'].shape[2] // 8)).cuda()
                    feat['reg'] = torch.from_numpy(gen_oracle_map(batch['reg'].detach().cpu().numpy(), 
                                                                  batch['ind'].detach().cpu().numpy(), 
                                                                  batch['inp'].shape[3] // 8, batch['inp'].shape[2] // 8)).cuda()
                else:               
                    feat = self.core(batch['inp'])[-1]
                    feat['hm'].sigmoid_()

                dets = ctdet_decode(feat['hm'], feat['wh'], reg=feat['reg'], K=100)
                dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[1])
                dets_out = ctdet_post_process(dets.copy(), batch['c'].cpu().numpy(), batch['s'].cpu().numpy(), 
                                               feat['hm'].shape[2], feat['hm'].shape[3], feat['hm'].shape[1])
                results[batch['img_id'][0]] = dets_out[0]
        cce = coco_eval(self.vdata.dataset.coco, results, self.cfg.OUTPUT_DIR)  

        logger.info('Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets={:>3d} ] = {:.3f}'.format(cce.params.maxDets[2], cce.stats[0]))
        logger.info('Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets={:>3d} ] = {:.3f}'.format(cce.params.maxDets[2], cce.stats[1]))
        logger.info('Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets={:>3d} ] = {:.3f}'.format(cce.params.maxDets[2], cce.stats[2]))
        logger.info('Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets={:>3d} ] = {:.3f}'.format(cce.params.maxDets[2], cce.stats[3]))
        logger.info('Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets={:>3d} ] = {:.3f}'.format(cce.params.maxDets[2], cce.stats[4]))
        logger.info('Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets={:>3d} ] = {:.3f}'.format(cce.params.maxDets[2], cce.stats[5]))
        logger.info('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets={:>3d} ] = {:.3f}'.format(cce.params.maxDets[0], cce.stats[6]))
        logger.info('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets={:>3d} ] = {:.3f}'.format(cce.params.maxDets[1], cce.stats[7]))
        logger.info('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets={:>3d} ] = {:.3f}'.format(cce.params.maxDets[2], cce.stats[8]))
        logger.info('Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets={:>3d} ] = {:.3f}'.format(cce.params.maxDets[2], cce.stats[9]))
        logger.info('Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets={:>3d} ] = {:.3f}'.format(cce.params.maxDets[2], cce.stats[10]))
        logger.info('Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets={:>3d} ] = {:.3f}'.format(cce.params.maxDets[2], cce.stats[11]))
        
        if not eval:
            self.accu = cce.stats[0]
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
                detection = {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": bbox_out,
                    "score": float("{:.2f}".format(score))
                }
                detections.append(detection)
    return detections

def coco_eval(coco, results, save_dir):
    json.dump(convert_eval_format(results, coco.getCatIds()), open('{}/results.json'.format(save_dir), 'w'))
    coco_dets = coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval
