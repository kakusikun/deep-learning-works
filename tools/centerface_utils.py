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
    transform_preds,
)

import math

def centerface_facial_target(cls_ids, bboxes, ptss, max_objs, num_classes, num_keypoints, outsize, **kwargs):
    '''
    According to CenterFace ( CenterFace: Joint Face Detection and Alignment Using Face as Point, https://arxiv.org/abs/1911.03599 ), 
    create the target for keypoints detection.

    Args:
        cls_ids (list): list of category of object.
        bboxes (list): list of 1x4 numpy arrays, the ground truth bounding box.
        ptss (list): list of a list with class of keypoints (int) and keypoints (Nx2 numpy array),
                     [pts1, pts2, ...].
        max_objs (int): the maximum number of objects in a image. To fix the object size, since that it's impossible to train model with dynamic size.
        num_classes (int): number of classes in dataset.
        num_keypoints (int): number of categories of keypoints in dataset.
        outsize (tuple): tuple of width and height of feature map of model output
    
    Returns:
        ret (dict): 
            hm (numpy.ndarray): shape Class x outsize H, outsize W, heat map which acts as the weight of object for training, 
                                the weight is a gaussian distribution with mean locate at the center of bounding box of objects in input data
            wh (numpy.ndarray): shape Object x 2(= width + height), width and height of objects in input data
            reg (numpy.ndarray): shape Object x 2(= width + height), offset of width and height of objects in input data, 
                                 since the width and height are integers
            reg_mask, ind (numpy.ndarray): shape Object, extract the entry to compute
            kps (numpy.ndarray): shape Object x (Keypoint Category x 2(= x_coord + y_coord)), the vector of keypoints with start from center 
                                 of the object and end to the keypoints
            kps_mask (numpy.ndarray): shape Object x (Keypoint Category x 2), to reduce memory of data usage for training
    '''
    output_w, output_h = outsize

    # center, face heatmap
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    # object size
    wh = np.zeros((max_objs, 2), dtype=np.float32)
    # landmark location relative to center and scaled by width and height of bounding box
    kps = np.zeros((max_objs, num_keypoints * 2), dtype=np.float32)
    # object offset
    reg = np.zeros((max_objs, 2), dtype=np.float32)       
    ind = np.zeros((max_objs), dtype=np.int64)
    reg_mask = np.zeros((max_objs), dtype=np.uint8)                       
    kps_mask = np.zeros((max_objs, num_keypoints * 2), dtype=np.uint8)

    draw_gaussian = draw_umich_gaussian

    for k, (cls_id, bbox, pts, valid_pts) in enumerate(zip(cls_ids, bboxes, ptss, valid_ptss)):
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]            
        if h > 0 and w > 0:
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_gaussian(hm[cls_id], ct_int, radius)
            """
            equation 4 in paper
            w = log(x2 - x1)
            h = log(y2 - y1)
            """

            wh[k] = np.log(1. * w), np.log(1. * h)
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
                        """
                        equation 5 in paper
                        lx = (lx - ctx) / w
                        ly = (ly - cty) / h
                        """
                        kps[k, j * 2: j * 2 + 2] = (pts[j, :2] - ct) / np.array([1. * w, 1. * h])
                        kps_mask[k, j * 2: j * 2 + 2] = 1
            draw_gaussian(hm[cls_id], ct_int, radius)
            
    ret = {'hm': hm, 'wh':wh, 'reg':reg,
           'reg_mask': reg_mask, 'ind': ind,
           'kps': kps, 'kps_mask': kps_mask}
    return ret

def centerface_decode(heat, wh, kps, reg=None, hm_kp=None, kp_reg=None, K=100):
    batch, _, _, _ = heat.size()
    num_joints = kps.shape[1] // 2
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
    wh = torch.exp(wh.view(batch, K, 2))

    kps = _tranpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_joints * 2)
    kps[..., ::2] *= wh[:,:,:1].expand(batch, K, num_joints)
    kps[..., 1::2] *= wh[:,:,1:].expand(batch, K, num_joints)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)

    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
  
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
        
    detections = torch.cat([bboxes, scores, kps, clses], dim=2)
      
    return detections

def centerface_post_process(dets, c, s, h, w, num_classes):
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

def centerface_bbox_target(cls_ids, bboxes, max_objs, num_classes, outsize, **kwargs):
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
            """
            equation 4 in paper
            w = log(x2 - x1)
            h = log(y2 - y1)
            """
            wh[k] = np.log(1. * w), np.log(1. * h)
            ind[k] = ct_int[1] * output_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1  
            
    ret = {'hm': hm, 'wh':wh, 'reg':reg,
           'reg_mask': reg_mask, 'ind': ind}
    return ret

if __name__ == '__main__':
    from config.config_factory import _C as cfg
    from database.data_factory import get_data
    from database.transform_factory import get_transform
    from database.dataset_factory import get_dataset
    from tools.oracle_utils import gen_oracle_map
    import logging
    logger = logging.getLogger("logger")
    import torch

    cfg.merge_from_file('/media/allen/mass/deep-learning-works/config/keypoint.yml')
    trans = get_transform(cfg, cfg.TRAIN_TRANSFORM)
    data = get_data(cfg.DB.DATA)(cfg)
    dataset = get_dataset(cfg.DB.DATASET)(data.train, trans, centerface_facial_target)

    img_map = {}
    for img_id, path in dataset.indice:
        img_map[img_id] = path

    batch = dataset[0]
    imgId = batch['img_id']
    img_path = img_map[imgId]
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].unsqueeze(0).numpy()
        else:
            try:
                batch[key] = batch[key][np.newaxis,:]
            except:
                continue

    
    feat = {}
    feat['hm'] = torch.from_numpy(batch['hm'])
    feat['wh'] = torch.from_numpy(gen_oracle_map(batch['wh'], batch['ind'], 128, 128))
    feat['reg'] = torch.from_numpy(gen_oracle_map(batch['reg'], batch['ind'], 128, 128))
    feat['kps'] = torch.from_numpy(gen_oracle_map(batch['kps'], batch['ind'], 128, 128))

    dets = centerface_decode(feat['hm'], feat['wh'], feat['kps'], reg=feat['reg'], K=100)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])                    
    dets_out = centerface_post_process(dets.copy(), batch['c'], batch['s'],
                                    feat['hm'].shape[2], feat['hm'].shape[3], feat['hm'].shape[1])
