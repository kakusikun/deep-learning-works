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

def centernet_keypoints_target(cls_ids, bboxes, ptss, max_objs, num_classes, num_keypoints, outsize, **kwargs):
    '''
    According to CenterNet ( Objects as Points, https://arxiv.org/abs/1904.07850 ), create the target for keypoints detection.

    Args:
        cls_ids (list): list of category of object.
        bboxes (list): list of 1x4 numpy arrays, the ground truth bounding box.
        ptss (list): list of a list with class of keypoints (int) and keypoints (Nx2 numpy array),
                     [pts1, pts2, ...].
        max_objs (int): the maximum number of objects in a image.
        num_classes (int): number of classes in dataset.
        num_keypoints (int): number of categories of keypoints in dataset.
        outsize (tuple): tuple of width and height of feature map of model output
    
    Returns:
        ret (dict): 
            hm (numpy.ndarray): Class x outsize H x outsize W, heat map which acts as the weight of object for training, 
                                the weight is a gaussian distribution with mean locate at the center of bounding box of objects in input data
            wh (numpy.ndarray): Object x 2(= width + height), width and height of objects in input data
            reg (numpy.ndarray): Object x 2(= width + height), offset of width and height of objects in input data, 
                                 since the width and height are integers
            reg_mask, ind (numpy.ndarray): Object, to reduce memory of data usage for training
            hm_kp (numpy.ndarray): Keypoint x outsize H, outsize W, heat map which acts as the weight of object for training, 
                                the weight is a gaussian distribution with mean locate at the keyponits in input data
            kps (numpy.ndarray): Object x (Keypoint x 2(= x_coord + y_coord)), the vector of keypoints with start from center 
                                 of the object and end to the keypoints
            kp_reg (numpy.ndarray): (Object x Keypoint) x 2(= width + height), offset of width and height of keypoints in input data, 
                                 since the width and height are integers
            kp_mask, kp_ind (numpy.ndarray): (Object x Keypoint), to reduce memory of data usage for training
            kps_mask (numpy.ndarray): (Object x Keypoint) x 2, to reduce memory of data usage for training
    '''
    output_w, output_h = outsize

    # center, object heatmap
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    # center, keypoint heatmap
    hm_kp = np.zeros((num_keypoints, output_h, output_w), dtype=np.float32)

    # object size
    wh = np.zeros((max_objs, 2), dtype=np.float32)
    # keypoint location relative to center
    kps = np.zeros((max_objs, num_keypoints * 2), dtype=np.float32)
    # object offset
    reg = np.zeros((max_objs, 2), dtype=np.float32)       
    ind = np.zeros((max_objs), dtype=np.int64)
    reg_mask = np.zeros((max_objs), dtype=np.uint8)                       
    kps_mask = np.zeros((max_objs, num_keypoints * 2), dtype=np.uint8)
    kp_reg = np.zeros((max_objs * num_keypoints, 2), dtype=np.float32)
    kp_ind = np.zeros((max_objs * num_keypoints), dtype=np.int64)
    kp_mask = np.zeros((max_objs * num_keypoints), dtype=np.int64)

    draw_gaussian = draw_umich_gaussian

    for k, (cls_id, bbox, pts, valid_pts) in enumerate(zip(cls_ids, bboxes, ptss, valid_ptss)):
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]            
        if h > 0 and w > 0:
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            wh[k] = 1. * w, 1. * h
            ind[k] = ct_int[1] * output_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1  
            num_kpts = pts[:,2].sum()
            if num_kpts == 0:
                hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
                reg_mask[k] = 0

            hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            hp_radius = max(0, int(hp_radius)) 
            for j in range(num_keypoints):
                if pts[j,2] > 0:
                    if pts[j, 0] >= 0 and pts[j, 0] < output_w and pts[j, 1] >= 0 and pts[j, 1] < output_h:
                        kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                        kps_mask[k, j * 2: j * 2 + 2] = 1
                        pt_int = pts[j, :2].astype(np.int32)
                        kp_reg[k * num_keypoints + j] = pts[j, :2] - pt_int
                        kp_ind[k * num_keypoints + j] = pt_int[1] * output_w + pt_int[0]
                        kp_mask[k * num_keypoints + j] = 1
                        draw_gaussian(hm_kp[j], pt_int, hp_radius)
            draw_gaussian(hm[cls_id], ct_int, radius)
            
    ret = {'hm': hm, 'wh':wh, 'reg':reg,
           'reg_mask': reg_mask, 'ind': ind,
           'hm_kp': hm_kp, 'kps': kps, 'kps_mask': kps_mask, 'kp_reg': kp_reg,
           'kp_ind': kp_ind, 'kp_mask': kp_mask}
    return ret

def centernet_pose_decode(heat, wh, kps, reg=None, hm_kp=None, kp_reg=None, K=100):
    batch, cat, height, width = heat.size()
    num_joints = kps.shape[1] // 2
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
  
    kps = _tranpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_joints * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
  
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    if hm_kp is not None:
        hm_kp = _nms(hm_kp)
        thresh = 0.1
        kps = kps.view(batch, K, num_joints, 2).permute(
            0, 2, 1, 3).contiguous() # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_kp, K=K) # b x J x K
        if kp_reg is not None:
            kp_reg = _tranpose_and_gather_feat(
                kp_reg, hm_inds.view(batch, -1))
            kp_reg = kp_reg.view(batch, num_joints, K, 2)
            hm_xs = hm_xs + kp_reg[:, :, :, 0]
            hm_ys = hm_ys + kp_reg[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5
          
        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_joints, K, K, 2)
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3) # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
            batch, num_joints, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)
        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
               (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
               (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_joints, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_joints * 2)
    detections = torch.cat([bboxes, scores, kps, clses], dim=2)
      
    return detections

def centernet_pose_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(dets[i, :, :2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
        dets[i, :, 5:-1] = transform_preds(dets[i, :, 5:-1].reshape(-1, 2), c[i], s[i], (w, h)).reshape(-1, dets.shape[2]-6)
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([dets[i, inds, :4].astype(np.float32),
                                            dets[i, inds, 4:5],
                                            dets[i, inds, 5:-1]], axis=1).tolist()
        ret.append(top_preds)
    return ret

def centernet_bbox_target(cls_ids, bboxes, max_objs, num_classes, outsize, **kwargs):
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
    output_w, output_h = outsize

    # center, object heatmap
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)

    # object size
    wh = np.zeros((max_objs, 2), dtype=np.float32)
    # object offset
    reg = np.zeros((max_objs, 2), dtype=np.float32)       
    ind = np.zeros((max_objs), dtype=np.int64)
    reg_mask = np.zeros((max_objs), dtype=np.uint8)                       

    draw_gaussian = draw_umich_gaussian

    for k, (cls_id, bbox) in enumerate(zip(cls_ids, bboxes)):
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]            
        if h > 0 and w > 0:
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_gaussian(hm[cls_id], ct_int, radius)
            wh[k] = 1. * w, 1. * h
            ind[k] = ct_int[1] * output_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1  
            
    ret = {'hm': hm, 'wh':wh, 'reg':reg,
           'reg_mask': reg_mask, 'ind': ind}
    return ret

def centernet_det_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
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
    if cat_spec_wh:
      wh = wh.view(batch, K, cat, 2)
      clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
      wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
      wh = wh.view(batch, K, 2)
    
    valid_object = wh[:,:,0].gt(0) * wh[:,:,1].gt(0)
    
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)[valid_object]
      
    return detections

def centernet_det_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret



