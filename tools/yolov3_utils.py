import numpy as np
import torch

def build_yolov3_JDE_targets(bboxes, ids, anchors, out_sizes):
    #TODO: build target for JDE which uses yolov3
    ID_THRESH = 0.5
    FG_THRESH = 0.5
    BG_THRESH = 0.4
    nB = len(target)  # number of images in batch
    assert(len(anchor_wh)==nA)

    tbox = torch.zeros(nB, nA, nGh, nGw, 4).cuda()  # batch size, anchors, grid size
    tconf = torch.LongTensor(nB, nA, nGh, nGw).fill_(0).cuda()
    tid = torch.LongTensor(nB, nA, nGh, nGw, 1).fill_(-1).cuda() 
    for b in range(nB):
        t = target[b]
        t_id = t[:, 1].clone().long().cuda()
        t = t[:,[0,2,3,4,5]]
        nTb = len(t)  # number of targets
        if nTb == 0:
            continue

        gxy, gwh = t[: , 1:3].clone() , t[:, 3:5].clone()
        gxy[:, 0] = gxy[:, 0] * nGw
        gxy[:, 1] = gxy[:, 1] * nGh
        gwh[:, 0] = gwh[:, 0] * nGw
        gwh[:, 1] = gwh[:, 1] * nGh
        gxy[:, 0] = torch.clamp(gxy[:, 0], min=0, max=nGw -1)
        gxy[:, 1] = torch.clamp(gxy[:, 1], min=0, max=nGh -1)

        gt_boxes = torch.cat([gxy, gwh], dim=1)                                            # Shape Ngx4 (xc, yc, w, h)
        
        anchor_mesh = generate_anchor(nGh, nGw, anchor_wh)
        anchor_list = anchor_mesh.permute(0,2,3,1).contiguous().view(-1, 4)              # Shpae (nA x nGh x nGw) x 4
        #print(anchor_list.shape, gt_boxes.shape)
        iou_pdist = bbox_iou(anchor_list, gt_boxes)                                      # Shape (nA x nGh x nGw) x Ng
        iou_max, max_gt_index = torch.max(iou_pdist, dim=1)                              # Shape (nA x nGh x nGw), both

        iou_map = iou_max.view(nA, nGh, nGw)       
        gt_index_map = max_gt_index.view(nA, nGh, nGw)

        #nms_map = pooling_nms(iou_map, 3)
        
        id_index = iou_map > ID_THRESH
        fg_index = iou_map > FG_THRESH                                                    
        bg_index = iou_map < BG_THRESH 
        ign_index = (iou_map < FG_THRESH) * (iou_map > BG_THRESH)
        tconf[b][fg_index] = 1
        tconf[b][bg_index] = 0
        tconf[b][ign_index] = -1

        gt_index = gt_index_map[fg_index]
        gt_box_list = gt_boxes[gt_index]
        gt_id_list = t_id[gt_index_map[id_index]]
        #print(gt_index.shape, gt_index_map[id_index].shape, gt_boxes.shape)
        if torch.sum(fg_index) > 0:
            tid[b][id_index] =  gt_id_list.unsqueeze(1)
            fg_anchor_list = anchor_list.view(nA, nGh, nGw, 4)[fg_index] 
            delta_target = encode_delta(gt_box_list, fg_anchor_list)
            tbox[b][fg_index] = delta_target
    return tconf, tbox, tid

def generate_anchor(nGh, nGw, anchor_wh):
    nA = len(anchor_wh)
    yy, xx =torch.meshgrid(torch.arange(nGh), torch.arange(nGw))
    xx, yy = xx.cuda(), yy.cuda()

    mesh = torch.stack([xx, yy], dim=0)                                              # Shape 2, nGh, nGw
    mesh = mesh.unsqueeze(0).repeat(nA,1,1,1).float()                                # Shape nA x 2 x nGh x nGw
    anchor_offset_mesh = anchor_wh.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, nGh,nGw) # Shape nA x 2 x nGh x nGw
    anchor_mesh = torch.cat([mesh, anchor_offset_mesh], dim=1)                       # Shape nA x 4 x nGh x nGw
    return anchor_mesh

def encode_delta(gt_box_list, fg_anchor_list):
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:,3]
    gx, gy, gw, gh = gt_box_list[:, 0], gt_box_list[:, 1], \
                     gt_box_list[:, 2], gt_box_list[:, 3]
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw/pw)
    dh = torch.log(gh/ph)
    return torch.stack([dx, dy, dw, dh], dim=1)