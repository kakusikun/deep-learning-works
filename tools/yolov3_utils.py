import numpy as np
import torch

def yolov3_JDE_targets(bboxes, ids, anchors, wh, strides, **kwargs):
    ID_THRESH = 0.5
    FG_THRESH = 0.5
    BG_THRESH = 0.4
    w, h = wh
    assert(len(anchors) % len(strides) == 0)
    n_a = len(anchors) // len(strides) // 2
    # normalized x1, y1, x2, y2
    gt_xyxy = torch.from_numpy(np.vstack(bboxes))
    ids = torch.LongTensor(ids)
    anchors = torch.Tensor(anchors).view(len(strides), -1, 2)
    rets = {}
    for stage, stride in enumerate(strides):
        _gt_xyxy = gt_xyxy.clone()
        g_w, g_h = w // stride, h // stride
        target_anchors = anchors[stage] / stride
        t_bbox = torch.zeros(n_a, g_h, g_w, 4)  # batch size, anchors, grid size
        t_conf = torch.LongTensor(n_a, g_h, g_w).fill_(0)
        t_pids = torch.LongTensor(g_h, g_w, 1).fill_(-1) 
        if len(bboxes) == 0:
            rets[(g_w, g_h)] = [t_bbox, t_conf, t_pids]
            continue

        # scaled x1, y1, x2, y2
        _gt_xyxy[:,[0, 2]] *= g_w
        _gt_xyxy[:,[1, 3]] *= g_h
        # xc, yc, w, h
        gt_xywh = xxyy2xcycwh(_gt_xyxy)
        gt_xywh[:,0] = torch.clamp(gt_xywh[:,0], min=0, max=g_w -1)
        gt_xywh[:,1] = torch.clamp(gt_xywh[:,1], min=0, max=g_h -1)
        
        anchor_mesh = generate_anchor(g_h, g_w, target_anchors)
        anchor_list = anchor_mesh.permute(0,2,3,1).contiguous().view(-1, 4)              # Shpae (n_a x g_h x g_w) x 4
        #print(anchor_list.shape, gt_boxes.shape)
        iou_pdist = bbox_iou(anchor_list, gt_xywh)                                      # Shape (n_a x g_h x g_w) x Ng
        iou_max, max_gt_index = torch.max(iou_pdist, dim=1)                             # Shape (n_a x g_h x g_w), both
        iou_map = iou_max.view(n_a, g_h, g_w)    
        gt_index_map = max_gt_index.view(n_a, g_h, g_w)
        id_map, id_index_map = iou_map.max(0)

        id_index = id_map > ID_THRESH
        fg_index = iou_map > FG_THRESH                                                    
        bg_index = iou_map < BG_THRESH 
        ign_index = (iou_map < FG_THRESH) * (iou_map > BG_THRESH)
        t_conf[fg_index] = 1
        t_conf[bg_index] = 0
        t_conf[ign_index] = -1
        gt_index = gt_index_map[fg_index]
        gt_box_list = gt_xywh[gt_index]
        gt_id_list = gt_index_map[:, id_index].gather(dim=0, index=id_index_map[id_index].view(1, -1))
        #print(gt_index.shape, gt_index_map[id_index].shape, gt_boxes.shape)
        if torch.sum(fg_index) > 0:
            t_pids[id_index] = ids[gt_id_list].view(-1, 1)
            fg_anchor_list = anchor_list.view(n_a, g_h, g_w, 4)[fg_index] 
            delta_target = encode_delta(gt_box_list, fg_anchor_list)
            t_bbox[fg_index] = delta_target
        rets[(g_w, g_h)] = {
            'bbox': t_bbox,
            'conf': t_conf,
            'pids': t_pids,
        }
    return rets


def xxyy2xcycwh(xxyy):
    xcycwh = torch.ones_like(xxyy)
    xcycwh[:,2:] = xxyy[:,2:] - xxyy[:,:2]
    xcycwh[:,:2] = xxyy[:,:2] + xcycwh[:,2:] / 2
    return xcycwh

def generate_anchor(g_h, g_w, anchor_wh):
    n_a = len(anchor_wh)
    yy, xx =torch.meshgrid(torch.arange(g_h), torch.arange(g_w))
    mesh = torch.stack([xx, yy], dim=0)                                              # Shape 2, g_h, g_w
    mesh = mesh.unsqueeze(0).repeat(n_a,1,1,1).float()                                # Shape n_a x 2 x g_h x g_w
    anchor_offset_mesh = anchor_wh.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, g_h,g_w) # Shape n_a x 2 x g_h x g_w
    anchor_mesh = torch.cat([mesh, anchor_offset_mesh], dim=1)                       # Shape n_a x 4 x g_h x g_w

    return anchor_mesh

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    N, M = len(box1), len(box2)
    # Transform from center and width to exact coordinates
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).view(-1,1).expand(N,M)
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).view(1,-1).expand(N,M)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def encode_delta(gt_box_list, fg_anchor_list):
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], fg_anchor_list[:, 2], fg_anchor_list[:,3]
    gx, gy, gw, gh = gt_box_list[:, 0], gt_box_list[:, 1], gt_box_list[:, 2] , gt_box_list[:, 3]
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw/pw)
    dh = torch.log(gh/ph)
    return torch.stack([dx, dy, dw, dh], dim=1)

def decode_delta(delta, fg_anchor_list):
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:,3]
    dx, dy, dw, dh = delta[:, 0], delta[:, 1], delta[:, 2], delta[:, 3]
    gx = pw * dx + px
    gy = ph * dy + py
    gw = pw * torch.exp(dw)
    gh = ph * torch.exp(dh)
    return torch.stack([gx, gy, gw, gh], dim=1)

def decode_delta_map(delta_map, anchors):
    '''
    :param: delta_map, shape (nB, nA, nGh, nGw, 4)
    :param: anchors, shape (nA,4)
    '''
    nB, nA, nGh, nGw, _ = delta_map.shape
    anchor_mesh = generate_anchor(nGh, nGw, anchors) 
    anchor_mesh = anchor_mesh.permute(0,2,3,1).contiguous()              # Shpae (nA x nGh x nGw) x 4
    anchor_mesh = anchor_mesh.unsqueeze(0).repeat(nB,1,1,1,1)
    pred_list = decode_delta(delta_map.view(-1,4), anchor_mesh.view(-1,4))
    pred_map = pred_list.view(nB, nA, nGh, nGw, 4)
    return pred_map

if __name__ == "__main__":
    from src.factory.config_factory import cfg, show_configs
    from src.factory.data_factory import DataFactory
    from src.factory.data_format_factory import DataFormatFactory
    from src.factory.transform_factory import TransformFactory
    from src.factory.loader_factory import LoaderFactory

    cfg.DB.PATH = "/media/allen/mass/DB"
    cfg.DB.DATA = "coco"
    cfg.DB.DATA_FORMAT = "coco"
    cfg.DB.LOADER = "coco"
    cfg.DB.USE_TRAIN = False
    cfg.DB.USE_TEST = True
    cfg.INPUT.SIZE = (512, 512)
    cfg.INPUT.TEST_BS = 4
    cfg.MODEL.STRIDES = [4]
    cfg.DB.TEST_TRANSFORM = "Resize Tensorize"
    cfg.REID.MSMT_ALL = False

    data = DataFactory.produce(cfg)
    trans = TransformFactory.produce(cfg, cfg.DB.TEST_TRANSFORM)
    dataset = DataFormatFactory.produce(cfg, data.val, trans)
    bboxes = dataset[0]['bboxes']
    ids = np.random.randint(0, 100, len(bboxes))
    anchors = [6,16, 8,23, 11,32, 16,45]
    build_yolov3_JDE_targets(bboxes, ids, anchors, [(128, 128)])