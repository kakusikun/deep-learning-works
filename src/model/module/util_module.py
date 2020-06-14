import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.module.base_module import ConvModule, SEModule, HSwish, get_ConvModule
from tools.utils import _tranpose_and_gather_feat

from tools.image import (
    get_affine_transform, 
    affine_transform, 
    draw_umich_gaussian, 
    draw_csp_gaussian,
    gaussian_radius, 
    color_aug,
)

class CenterNetBBoxModule(nn.Module):
    def __init__(self, HMLoss, WHLoss, RegLoss):
        super(CenterNetBBoxModule, self).__init__()

    def forward(self, hm, wh, reg, target):
        gt_bboxes = target['bboxes']
        n, _, h, w = hm.size()
        _, nobj, _ = gt_bboxes.size()
        # N x Obj x 4, recover to size of feature map
        bboxes = torch.ones_like(gt_bboxes)[..., :4]
        bboxes[..., [0, 2]] = gt_bboxes[..., [0, 2]] * w
        bboxes[..., [1, 3]] = gt_bboxes[..., [1, 3]] * h
        # N x Obj, get width and height
        bw, bh = bboxes[..., 2] - bboxes[..., 0], bboxes[..., 3] - bboxes[..., 1]
        radius = gaussian_radius(bw.view(-1), bh.view(-1)).view_as(bw)
        ct_xs = bboxes[..., [0, 2]].sum().div(2)
        ct_ys = bboxes[..., [1, 3]].sum().div(2)
        # N x Obj x 3
        cts = torch.stack([ct_xs, ct_ys, radius], dim=-1)
        cts_int = cts.int()

        # N x Obj, true bbox having height and width
        valid_bboxes_mask = gt_bboxes[...,-1].eq(1) * bw.gt(0) * bh.gt(0)
        num_boxes = valid_bboxes_mask.sum(dim=0)
        gt_hm = torch.zeros_like(hm)
        ind = torch.zeros_like(valid_bboxes_mask)
        ind[valid_bboxes_mask] = cts_int[valid_bboxes_mask][:2] * w + cts_int[valid_bboxes_mask][:2]
        for ni in range(n):
            mask = valid_bboxes_mask[ni]
            for ct_int in cts_int[ni][mask]:
                draw_umich_gaussian(gt_hm[ni].numpy(), ct_int[:2].numpy(), ct_int[2])


def gaussian_radius(w, h, min_overlap=0.7):
    a1  = 1
    b1  = (h + w)
    c1  = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (h + w)
    c2  = (1 - min_overlap) * w * h
    sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (h + w)
    c3  = (min_overlap - 1) * w * h
    sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
    r3  = (b3 + sq3) / 2
    r = torch.stack([r1, r2, r3], dim=1).min(dim=1)[0]
    return r
