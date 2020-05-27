from src.engine import *
from tools.oracle_utils import gen_oracle_map
from tools.centernet_utils import centernet_det_post_process
from tools.scopehead_utils import scopehead_det_decode
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import json
from collections import defaultdict
# recover = T.Compose([T.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225], std = [1/0.229,1/0.224,1/0.225])])

class Shufflenetv2ScopeHeadOD(BaseEngine):
    def __init__(self, cfg, graph, loader, solvers, visualizer):
        super(Shufflenetv2ScopeHeadOD, self).__init__(cfg, graph, loader, solvers, visualizer)
        w, h = cfg.INPUT.SIZE
        self.out_sizes = [(w // s, h // s) for s in cfg.MODEL.STRIDES]
    def _train_once(self):
        for batch in tqdm(self.tdata, desc=f"TRAIN[{self.epoch}/{self.cfg.SOLVER.MAX_EPOCHS}]"):
            self._train_iter_start()
            if self.use_gpu:
                for key in batch:
                    if isinstance(batch[key], dict):
                        for sub_key in batch[key]:
                            if self.cfg.DISTRIBUTED:
                                batch[key][sub_key] = batch[key][sub_key].to(self.device, non_blocking=True)
                            else:
                                batch[key][sub_key] = batch[key][sub_key].cuda()
                    elif not isinstance(batch[key], torch.Tensor):
                        continue
                    else:
                        if self.cfg.DISTRIBUTED:
                            batch[key] = batch[key].to(self.device, non_blocking=True)
                        else:
                            batch[key] = batch[key].cuda()
            outputs = self.graph.run(batch['inp'])
            self.loss, self.losses = self.graph.loss_head(outputs, batch)
            self._train_iter_end()

    def _evaluate(self, eval=False):
        logger.info("Epoch {} evaluation start".format(self.epoch))
        title = "EVALUATE" if eval else f"TEST[{self.epoch}/{self.cfg.SOLVER.MAX_EPOCHS}]"
        results = {}
        self._eval_epoch_start()
        with torch.no_grad():
            self._eval_epoch_start()
            for batch in tqdm(self.vdata, desc=title): 
                if self.use_gpu:
                    for key in batch:
                        if isinstance(batch[key], dict):
                            for sub_key in batch[key]:
                                if self.cfg.DISTRIBUTED:
                                    batch[key][sub_key] = batch[key][sub_key].to(self.device, non_blocking=True)
                                else:
                                    batch[key][sub_key] = batch[key][sub_key].cuda()
                        elif not isinstance(batch[key], torch.Tensor):
                            continue
                        else:
                            if self.cfg.DISTRIBUTED:
                                batch[key] = batch[key].to(self.device, non_blocking=True)
                            else:
                                batch[key] = batch[key].cuda()
                dets_outs = defaultdict(list)
                for out_size in self.out_sizes:
                    if self.cfg.ORACLE:
                        feat = {}
                        feat['hm']  = batch[out_size]['hm']
                        feat['wh']  = torch.from_numpy(
                            gen_oracle_map(
                                batch[out_size]['wh'].detach().cpu().numpy(), 
                                batch[out_size]['ind'].detach().cpu().numpy(), 
                                batch['inp'].shape[3] // self.cfg.MODEL.STRIDES[0], 
                                batch['inp'].shape[2] // self.cfg.MODEL.STRIDES[0]
                            )
                        )
                        feat['reg'] = torch.from_numpy(
                            gen_oracle_map(
                                batch[out_size]['reg'].detach().cpu().numpy(), 
                                batch[out_size]['ind'].detach().cpu().numpy(), 
                                batch['inp'].shape[3] // self.cfg.MODEL.STRIDES[0], 
                                batch['inp'].shape[2] // self.cfg.MODEL.STRIDES[0]
                            )
                        )
                        if self.use_gpu:
                            feat['wh'] = feat['wh'].cuda()
                            feat['reg'] = feat['reg'].cuda()
                    else:               
                        feat = self.graph.run(batch['inp'])[out_size]
                        feat['hm'].sigmoid_()
                    
                    if self.cfg.DB.TARGET_FORMAT == 'centerface_bbox':
                        feat['wh'].exp_()

                    dets = scopehead_det_decode(feat['hm'], feat['wh'], reg=feat['reg'], K=100)
                    dets = dets.detach().cpu().numpy().reshape(1, 100, -1)
                    dets_out = centernet_det_post_process(
                        dets.copy(), 
                        batch['c'].cpu().numpy(), 
                        batch['s'].cpu().numpy(), 
                        feat['hm'].shape[2], 
                        feat['hm'].shape[3], 
                        feat['hm'].shape[1]
                    )[0]
                    for cat in dets_out:
                        dets_outs[cat].extend(dets_out[cat])
                results[batch['img_id'][0]] = dets_outs

        cce = coco_eval(self.vdata.dataset.coco[0], results, self.cfg.OUTPUT_DIR)  

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
    for image_id in tqdm(all_bboxes, desc="COCO EVAL"):
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
