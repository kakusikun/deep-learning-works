import numpy as np
from tools.image import get_affine_transform, affine_transform, draw_umich_gaussian, gaussian_radius, color_aug
import math

def CenterNet_keypoints_target(bboxes, ptss, valid_ptss, max_objs, num_classes, num_joints, outsize):
    '''
    According to CenterNet ( Objects as Points, https://arxiv.org/abs/1904.07850 ), create the target for keypoints detection.

    Args:
        bboxes (list): list of 1x4 numpy arrays, the ground truth bounding box.
        ptss (list): list of a list with class of keypoints (int) and keypoints (Nx2 numpy array),
                     [[c1, pts1], [c2, pts2], ...].
        valid_ptss (list): list of 1xN numpy arrays where the N is equal to the N of pts in ptss, indicating the visibility of each pt in pts.
                    2 is visible, 1 is occlusion and 0 is not labeled.
        max_objs (int): the maximum number of objects in a image.
        num_classes (int): number of classes in dataset.
        num_joints (int): number of categories of keypoints in dataset.
        outsize (tuple): tuple of width and height of feature map of model output
    
    Returns:
        ret (dict): 
            hm (numpy.ndarray): shape Class x outsize H, outsize W, heat map which acts as the weight of object for training, 
                                the weight is a gaussian distribution with mean locate at the center of bounding box of objects in input data
            wh (numpy.ndarray): shape Object x 2(= width + height), width and height of objects in input data
            reg (numpy.ndarray): shape Object x 2(= width + height), offset of width and height of objects in input data, 
                                 since the width and height are integers
            reg_mask, ind (numpy.ndarray): shape Object, to reduce memory of data usage for training
            hm_kp (numpy.ndarray): shape (Keypoint Category) x outsize H, outsize W, heat map which acts as the weight of object for training, 
                                the weight is a gaussian distribution with mean locate at the keyponits in input data
            kps (numpy.ndarray): shape Object x (Keypoint Category x 2(= x_coord + y_coord)), the vector of keypoints with start from center 
                                 of the object and end to the keypoints
            kp_reg (numpy.ndarray): shape (Object x Keypoint Category) x 2(= width + height), offset of width and height of keypoints in input data, 
                                 since the width and height are integers
            kp_mask, kp_ind (numpy.ndarray): shape (Object x Keypoint Category), to reduce memory of data usage for training
            kps_mask (numpy.ndarray): shape (Object x Keypoint Category) x 2, to reduce memory of data usage for training
    '''
    output_w, output_h = outsize

    # center, object heatmap
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    # center, keypoint heatmap
    hm_kp = np.zeros((num_joints, output_h, output_w), dtype=np.float32)

    # object size
    wh = np.zeros((max_objs, 2), dtype=np.float32)
    # keypoint location relative to center
    kps = np.zeros((max_objs, num_joints * 2), dtype=np.float32)
    # object offset
    reg = np.zeros((max_objs, 2), dtype=np.float32)       
    ind = np.zeros((max_objs), dtype=np.int64)
    reg_mask = np.zeros((max_objs), dtype=np.uint8)                       
    kps_mask = np.zeros((max_objs, num_joints * 2), dtype=np.uint8)
    kp_reg = np.zeros((max_objs * num_joints, 2), dtype=np.float32)
    kp_ind = np.zeros((max_objs * num_joints), dtype=np.int64)
    kp_mask = np.zeros((max_objs * num_joints), dtype=np.int64)

    draw_gaussian = draw_umich_gaussian

    for k, (bbox, (cls_id, pts), valid_pts) in enumerate(zip(bboxes, ptss, valid_ptss)):
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
                        kp_reg[k * num_joints + j] = pts[j, :2] - pt_int
                        kp_ind[k * num_joints + j] = pt_int[1] * output_w + pt_int[0]
                        kp_mask[k * num_joints + j] = 1
                        draw_gaussian(hm_kp[j], pt_int, hp_radius)
            draw_gaussian(hm[cls_id], ct_int, radius)
            
    ret = {'hm': hm, 'wh':wh, 'reg':reg,
           'reg_mask': reg_mask, 'ind': ind,
           'hm_kp': hm_kp, 'kps': kps, 'kps_mask': kps_mask, 'kp_reg': kp_reg,
           'kp_ind': kp_ind, 'kp_mask': kp_mask}
    return ret

