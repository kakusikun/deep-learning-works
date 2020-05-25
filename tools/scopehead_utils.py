import numpy as np
import torch
from tools.image import (
    get_affine_transform, 
    affine_transform, 
    draw_umich_gaussian, 
    gaussian_radius, 
    color_aug,
)

from tools.utils import (
    _tranpose_and_gather_feat,
    _nms,
    _topk,
    _topk_channel,
    transform_preds,
)

import math

def scopehead_bbox_target(cls_ids, bboxes, ids, max_objs, num_classes, out_sizes, num_bins=5, **kwargs):
    '''
    According to CenterNet ( Objects as Points, https://arxiv.org/abs/1904.07850 ), create the target for object detection.

    Using the concept about anchor-based approach from Scope Head ( https://arxiv.org/abs/2005.04854 )

    Since the learning of 1D anchor scale in Scope Head is ambiguous, using the size of feature instead

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
            wh (numpy.ndarray): Object x 4(= left, up, right, down) x Bins, probabilities of bins for each direction
            reg (numpy.ndarray): Object x 4(= left, up, right, down), scale normalized by unit length of each direction
                                 since the width and height are integers
            reg_mask, ind (numpy.ndarray): Object, to reduce memory of data usage for training
    '''
    rets = {}
    for output_w, output_h in out_sizes:
        unit_w = output_w // num_bins
        unit_h = output_h // num_bins

        # center, object heatmap
        hm = torch.zeros(num_classes, output_h, output_w)
        # 4 directions, left, up, right, down, since the center of object is searched, the symmetric bbox is assumed
        bins = torch.zeros(max_objs, 4).long()
        # 4 directions, adjust the length of selected bin
        reg = torch.zeros(max_objs, 4)       
        ind = torch.zeros(max_objs).long()
        reg_mask = torch.zeros(max_objs).byte()    
        pids = torch.ones(max_objs).long() * -1

        draw_gaussian = draw_umich_gaussian

        for k, (cls_id, _bbox, pid) in enumerate(zip(cls_ids, bboxes, ids)):
            bbox = _bbox.copy()
            bbox[[0, 2]] *= output_w
            bbox[[1, 3]] *= output_h

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]            
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = torch.FloatTensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                ct_int = ct.int()
                draw_gaussian(hm[cls_id].numpy(), ct_int.numpy(), radius)
                bins[k][0] = bbox[0] // unit_w
                bins[k][1] = bbox[1] // unit_h
                bins[k][2] = (output_w - bbox[2]) // unit_w
                bins[k][3] = (output_h - bbox[3]) // unit_h
                reg[k][0] = (bbox[0] % unit_w) / unit_w
                reg[k][1] = (bbox[1] % unit_h) / unit_h
                reg[k][2] = ((output_w-bbox[2]) % unit_w) / unit_w
                reg[k][3] = ((output_h-bbox[3]) % unit_h) / unit_h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg_mask[k] = 1  
                pids[k] = pid
                
        rets[(output_w, output_h)] = {
            'hm': hm,
            'wh': bins,
            'reg': reg,
            'reg_mask': reg_mask,
            'ind': ind,
            'pids': pids
        }

    return rets

def scopehead_det_decode(heat, wh, reg, K=100, num_bins=5):
    batch, cat, height, width = heat.size()
    unit = torch.Tensor([width // num_bins, height // num_bins, width // num_bins, height // num_bins])
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    reg = _tranpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 4)
    wh = _tranpose_and_gather_feat(wh, inds)
    wh = (wh.view(batch, K, 4, -1).max(dim=-1)[1] + reg) * unit
    
    valid_object = wh[:,:,0].gt(0) * wh[:,:,1].gt(0)
    
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1], 
                        ys - wh[..., 1:2],
                        xs + wh[..., 2:3], 
                        ys + wh[..., 3:4]], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)[valid_object]
      
    return detections



