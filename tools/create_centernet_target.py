import numpy as np
from tools.image import get_affine_transform, affine_transform, draw_umich_gaussian, gaussian_radius, color_aug
import math

def keypoints_target(bboxes, ptss, valid_ptss, max_objs -> int, num_classes -> int, num_joints -> int, outsize -> tuple):
    output_w, output_h = outsize

    # center, object heatmap
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    # center, keypoint heatmap
    hm_hp = np.zeros((num_joints, output_h, output_w), dtype=np.float32)

    # object size
    wh             = np.zeros((max_objs, 2), dtype=np.float32)
    # keypoint location relative to center
    kps = np.zeros((max_objs, num_joints * 2), dtype=np.float32)
    # object offset
    reg            = np.zeros((max_objs             , 2             ), dtype=np.float32)       
    ind            = np.zeros((max_objs             ), dtype=np.int64)
    reg_mask       = np.zeros((max_objs             ), dtype=np.uint8)                       
    kps_mask       = np.zeros((max_objs             , num_joints * 2), dtype=np.uint8)
    hp_offset      = np.zeros((max_objs * num_joints, 2), dtype=np.float32)
    hp_ind         = np.zeros((max_objs * num_joints), dtype=np.int64)
    hp_mask        = np.zeros((max_objs * num_joints), dtype=np.int64)

    draw_gaussian = draw_umich_gaussian

    for bbox, (cls_id, pts), valid_pts in zip(bboxes, ptss, valid_ptss):
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
            num_kpts = valid_pts.sum()
            if num_kpts == 0:
                hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
                reg_mask[k] = 0

            hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            hp_radius = max(0, int(hp_radius)) 
            for j in range(num_joints):
                if valid_pts[j] > 0:
                    if pts[j, 0] >= 0 and pts[j, 0] < output_w and pts[j, 1] >= 0 and pts[j, 1] < output_h:
                        kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                        kps_mask[k, j * 2: j * 2 + 2] = 1
                        pt_int = pts[j, :2].astype(np.int32)
                        hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
                        hp_ind[k * num_joints + j] = pt_int[1] * output_w + pt_int[0]
                        hp_mask[k * num_joints + j] = 1
                        draw_gaussian(hm_hp[j], pt_int, hp_radius)
            draw_gaussian(hm[cls_id], ct_int, radius)
            
    ret = {'inp': inp,
            'hm': hm, 'wh':wh, 'reg':reg,
            'reg_mask': reg_mask, 'ind': ind,
            'hm_hp': hm_hp, 'hps': kps, 'hps_mask': kps_mask, 'hp_reg': hp_offset,
            'hp_ind': hp_ind, 'hp_mask': hp_mask,
            'img_id': img_id, 'c': c, 's': s}
    return ret

