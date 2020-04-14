from src.graph import *
from src.model.backbone.hacnn import ShuffleB

class _Model(nn.Module):
    def __init__(self, cfg):
        super(_Model, self).__init__()
        self.backbone = BackboneFactory.produce(cfg)
        self.local_branch = HarmAttn(
            block=ShuffleB, 
            n_stream=4, 
            channels=self.backbone.stage_out_channels
        )
        self.local_head = ReIDL2Head(cfg.MODEL.FEATSIZE, cfg.REID.NUM_PERSON)
        self.global_head = ReIDL2Head(cfg.MODEL.FEATSIZE, cfg.REID.NUM_PERSON)

    def forward(self, x):
        stages = self.backbone(x)
        global_feat, _ = stages[-1]
        local_feat = self.local_branch(stages[:-1])
        l2_global_feat, id_global_feat = self.global_head(global_feat)
        l2_local_feat, id_local_feat = self.local_head(local_feat)
        outputs = {
            'g_l2': l2_global_feat,
            'g_id': id_global_feat,
            'l_l2': l2_local_feat,
            'l_id': id_local_feat,
            'embb': torch.cat([l2_local_feat, l2_global_feat], dim=1) if not self.training else None
        }
        return outputs
        

class HarmAttenReID(BaseGraph):
    def __init__(self, cfg):
        super(HarmAttenReID, self).__init__(cfg)
    
    def build(self):
        self.model = _Model(self.cfg)
        self.crit = {}
        self.crit['g_cels'] = CrossEntropyLossLS(self.cfg.REID.NUM_PERSON)
        self.crit['g_triplet'] = SoftTripletLoss()
        self.crit['l_cels'] = CrossEntropyLossLS(self.cfg.REID.NUM_PERSON)
        self.crit['l_triplet'] = SoftTripletLoss()

        def loss_head(outputs, batch):
            assert outputs['g_id'] is not None and outputs['l_id'] is not None
            losses = {
                'g_cels':self.crit['g_cels'](outputs['g_id'], batch['pid']), 
                'g_triplet':self.crit['g_triplet'](outputs['g_l2'], batch['pid']), 
                'l_cels':self.crit['l_cels'](outputs['l_id'], batch['pid']), 
                'l_triplet':self.crit['l_triplet'](outputs['l_l2'], batch['pid']), 
            }
            loss = losses['g_cels'] + losses['g_triplet'] + losses['l_cels'] + losses['l_triplet']
            return loss, losses

        self.loss_head = loss_head
