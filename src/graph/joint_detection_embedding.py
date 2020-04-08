from src.graph import *

class _Model(nn.Module):
    def __init__(self, cfg):
        super(_Model, self).__init__()
        assert(len(cfg.YOLO.ANCHORS) % len(cfg.MODEL.STRIDES) == 0)
        n_anchors_per_head = len(cfg.YOLO.ANCHORS) // len(cfg.MODEL.STRIDES)
        self.backbone = BackboneFactory.produce(cfg)
        self.fpn = FPN(
            configs=["i_0", "i_1", "i_2"], 
            incs=self.backbone.stage_out_channels[-3:],
            oss=[1,2,4],
            oucs=self.backbone.stage_out_channels[-3:]
        )
        self.prediction_heads = nn.ModuleList()
        for inc in self.backbone.stage_out_channels[-3:]:
            self.prediction_heads.append(YOLOv3PredictionHead(inc, n_anchors_per_head * 6, cfg.MODEL.FEATSIZE))
    
    def forward(self, x):
        ps = self.backbone(x)
        ps = self.fpn(ps)
        for i in range(len(ps)):
            ps[i] = self.prediction_heads[i][ps[i]]
        return ps

class _LossHead(nn.Module):
    def __init__(self):
        super(_LossHead, self).__init__()
        self.crit = {}
        self.crit['bbox'] = nn.SmoothL1Loss() 
        self.crit['conf'] = nn.CrossEntropyLoss()
        self.crit['reg'] = L1Loss()

    def forward(self, feats, batch):
        bbox_loss = 0.0
        conf_loss = 0.0
        id_loss = 0.0

        for feat in feats:
            p, p_emb = feat[:, :24, ...], feat[:, 24:, ...]
            g_h, g_w = p.shape[-2:]
            t_bbox = batch[f"yolov3_{g_w}x{g_h}_t_bbox"] # N x A x H x W x 4
            t_conf = batch[f"yolov3_{g_w}x{g_h}_t_conf"] # N x A x H x W
            t_pids = batch[f"yolov3_{g_w}x{g_h}_t_pids"] # N x A x H x W x 1 
            n_a = t_bbox.shape[1]

            # N x A x H x W x 6
            p = p.view(-1, n_a, 24 // n_a, g_h, g_w).permute(0, 1, 3, 4, 2).contiguous()

            # N x H x W x D 
            p_emb = p_emb.permute(0,2,3,1).contiguous()
            # N x A x H x W x 4
            p_bbox = p[..., :4]
            # N x 2 x A x H x W
            p_conf = p[..., 4:6].permute(0, 4, 1, 2, 3)  # Conf

            # N x A x H x W
            mask = t_conf > 0

            n_valid = mask.sum().float()
            if n_valid > 0:
                bbox_loss += self.crit['bbox'](p_bbox[mask], t_bbox[mask])

            conf_loss += self.crit['conf'](p_conf, t_conf)

            # N x H x W
            emb_mask, _ = mask.max(1)
            
            # For convenience we use max(1) to decide the id, TODO: more reseanable strategy
            # N x H x W x 1
            t_pids, _ = t_pids.max(1) 
            # ? x 1
            t_pids = t_pids[emb_mask]
            # ? x D
            embedding = p_emb[emb_mask].contiguous()
            embedding = self.emb_scale * F.normalize(embedding)
            
            if len(embedding) > 1:
                logits = classifier(embedding).contiguous()
                lid =  self.IDLoss(logits, tids.squeeze())

            # Sum loss components
            loss = torch.exp(-self.s_r)*lbox + torch.exp(-self.s_c)*lconf + torch.exp(-self.s_id)*lid + \
                   (self.s_r + self.s_c + self.s_id)
            loss *= 0.5

            return loss, loss.item(), lbox.item(), lconf.item(), lid.item(), nT

        else:
            p_conf = torch.softmax(p_conf, dim=1)[:,1,...].unsqueeze(-1)
            p_emb = F.normalize(p_emb.unsqueeze(1).repeat(1,self.nA,1,1,1).contiguous(), dim=-1)
            #p_emb_up = F.normalize(shift_tensor_vertically(p_emb, -self.shift[self.layer]), dim=-1)
            #p_emb_down = F.normalize(shift_tensor_vertically(p_emb, self.shift[self.layer]), dim=-1)
            p_cls = torch.zeros(nB,self.nA,nGh,nGw,1).cuda()               # Temp
            p = torch.cat([p_box, p_conf, p_cls, p_emb], dim=-1)
            #p = torch.cat([p_box, p_conf, p_cls, p_emb, p_emb_up, p_emb_down], dim=-1)
            p[..., :4] = decode_delta_map(p[..., :4], self.anchor_vec.to(p))
            p[..., :4] *= self.stride

            return p.view(nB, -1, p.shape[-1])


class CenterNetObjectDetection(BaseGraph):
    def __init__(self, cfg):
        super(CenterNetObjectDetection, self).__init__(cfg)        
    
    def build(self):
        self.model = _Model(self.cfg)     
        self.loss_head = _LossHead()
