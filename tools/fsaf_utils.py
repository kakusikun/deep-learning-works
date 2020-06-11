from itertools import product

import numpy as np
import torch

from tools.utils import (
    _tranpose_and_gather_feat,
    _nms,
    _topk,
)

EFFECTIVE = 0.2
IGNORE = 0.5

def fsaf_bbox_target(cls_ids, bboxes, ids, max_objs, num_classes, out_sizes, **kwargs):
    '''
    According to CenterNet ( Objects as Points, https://arxiv.org/abs/1904.07850 ), create the target for object detection.

    Args:
        cls_ids (list): list of category of object.
        bboxes (list): list of 1x4 numpy arrays, the ground truth bounding box.
        max_objs (int): the maximum number of objects in a image.
        num_classes (int): number of classes in dataset.
        outsize (tuple): tuple of width and height of feature map of model output
    
    Returns:
        ret (dict): 
            hm (numpy.ndarray): Class x outsize H x outsize W, heat map which acts as the weight of object for training, 
                                the weight is a gaussian distribution with mean locate at the center of bounding box of objects in input data
            wh (numpy.ndarray): Object x 2(= width + height), width and height of objects in input data
            reg (numpy.ndarray): Object x 2(= width + height), offset of width and height of objects in input data, 
                                 since the width and height are integers
            reg_mask, ind (numpy.ndarray): Object, to reduce memory of data usage for training
    '''
    rets = {}
    _bboxes = np.array(bboxes)
    area = (_bboxes[:,2] - _bboxes[:,0]) * (_bboxes[:,3] - _bboxes[:,1])
    build_order = np.argsort(area)[::-1]
    max_effetive = int(out_sizes[0][0] * out_sizes[0][1] / 2)
    for output_w, output_h in out_sizes:
        # center, object heatmap
        hm = torch.zeros(num_classes, output_h, output_w)
        ind = torch.zeros(max_objs*max_effetive).long()
        ecount = torch.zeros(max_objs).long()
        mask = torch.zeros(max_objs*max_effetive).byte()

        idx = 0
        for k in build_order:
            cls_id = cls_ids[k]
            bbox = bboxes[k].copy()
            pid = ids[k]
            bbox[[0, 2]] *= output_w
            bbox[[1, 3]] *= output_h

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]            
            if h > 0 and w > 0:
                # effective radius
                e_w, e_h = int(w * EFFECTIVE / 2), int(h * EFFECTIVE / 2)
                i_w, i_h = int(w * IGNORE / 2), int(h * IGNORE / 2)
                ct = torch.FloatTensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                ct_int = ct.int()
                hm[:, (ct_int[1]-i_h):(ct_int[1]+i_h+1), (ct_int[0]-i_w):(ct_int[0]+i_w+1)] = -1
                hm[:, (ct_int[1]-e_h):(ct_int[1]+e_h+1), (ct_int[0]-e_w):(ct_int[0]+e_w+1)] = 1
                for ei, (ex, ey) in enumerate(product(range(ct_int[0]-i_w, ct_int[0]+i_w+1), range(ct_int[1]-i_h, ct_int[1]+i_h+1))):
                    ind[idx] = ey * output_w + ex
                    mask[idx] = 1
                    idx += 1
                ecount[k] = ei + 1
            else:
                bboxes[k][-1] = 0.0

        rets[(output_w, output_h)] = {
            'hm': hm,
            'mask': mask,
            'ecount': ecount,
            'ind': ind,
        }

    return rets

def fsaf_det_decode(heat, wh, reg=None, K=100):
    batch, cat, height, width = heat.size()
    
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
      reg = _tranpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 4)
    
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1], 
                        ys - wh[..., 1:2],
                        xs + wh[..., 2:3], 
                        ys + wh[..., 3:4]], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
      
    return detections