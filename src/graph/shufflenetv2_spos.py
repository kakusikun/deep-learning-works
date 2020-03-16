from sys import maxsize
from src.graph import *
from src.model.backbone.shufflenet_oneshot import ShuffleNetOneShot
from src.model.module.loss_module import CrossEntropyLossLS
import random

class _Model(nn.Module):
    def __init__(self, cfg):
        self.backbone = ShuffleNetOneShot(
            strides=[1, 1, 2, 2, 2],
            stage_repeats=[4, 8, 4],
            stage_out_channels=[24, 116, 232, 464],
            mode='v2'
        )
        self.head = ClassifierHead(cfg.MODEL.FEATSIZE, cfg.DB.NUM_CLASSES)
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

class SPOS(BaseGraph):
    def __init__(self, cfg):
        self.strides = [2, 2, 2]
        self.stage_repeats = [8,8,8]
        self.stage_out_channels = [64, 160, 320]
        self.candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        self.last_conv_out_channel = 512  
        self._generate_block_candidates()
        super().__init__(cfg)

    def build(self):
        self.model = _Model(self.cfg)
    
        self.crit = {}
        self.crit['cels'] = CrossEntropyLossLS(self.cfg.DB.NUM_CLASSES)

        def loss_head(feat, batch):
            losses = {'cels':self.crit['cels'](feat, batch['target'])}
            loss = losses['cels']
            return loss, losses

        self.loss_head = loss_head

    def _generate_block_candidates(self):
        self.block_choices = []
        for num_repeats in self.stage_repeats:
            for i in range(num_repeats):
                if i == 0:
                    self.block_choices.append([1,2])
                else:
                    self.block_choices.append([0,1,2])

    def random_block_choices(self):
        block_number = sum(self.stage_repeats)
        block_choices = []
        for i in range(block_number):
            block_choices.append(random.choice(self.block_choices[i]))
        return block_choices

    def random_channel_choices(self, epoch_after_cs=maxsize):
        """
        candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        mode: str, "dense" or "sparse". Sparse mode select # channel from candidate scales. Dense mode selects
              # channels between randint(min_channel, max_channel).
        """
        assert len(self.stage_repeats) == len(self.stage_out_channels)
        # From [1.0, 1.2, 1.4, 1.6, 1.8, 2.0] to [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], warm-up stages are
        # not just 1 epoch, but 2, 3, 4, 5 accordingly.

        epoch_delay_early = {0: 0,  # 8
                             1: 1, 2: 1,  # 7
                             3: 2, 4: 2, 5: 2,  # 6
                             6: 3, 7: 3, 8: 3, 9: 3,  # 5
                             10: 4, 11: 4, 12: 4, 13: 4, 14: 4,
                             15: 5, 16: 5, 17: 5, 18: 5, 19: 5, 20: 5,
                             21: 6, 22: 6, 23: 6, 24: 6, 25: 6, 27: 6, 28: 6,
                             29: 6, 30: 6, 31: 6, 32: 6, 33: 6, 34: 6, 35: 6, 36: 7,
                           }
        epoch_delay_late = {0: 0,
                            1: 1,
                            2: 2,
                            3: 3,
                            4: 4, 5: 4,  # warm up epoch: 2 [1.0, 1.2, ... 1.8, 2.0]
                            6: 5, 7: 5, 8: 5,  # warm up epoch: 3 ...
                            9: 6, 10: 6, 11: 6, 12: 6,  # warm up epoch: 4 ...
                            13: 7, 14: 7, 15: 7, 16: 7, 17: 7,  # warm up epoch: 5 [0.4, 0.6, ... 1.8, 2.0]
                            18: 8, 19: 8, 20: 8, 21: 8, 22: 8, 23: 8}  # warm up epoch: 6, after 17, use all scales
        select_all_channels = False
        if epoch_after_cs < 0:
            select_all_channels = True
        else:
            if 0 <= epoch_after_cs <= 23 and self.stage_out_channels[0] >= 64:
                delayed_epoch_after_cs = epoch_delay_late[epoch_after_cs]
            elif 0 <= epoch_after_cs <= 36 and self.stage_out_channels[0] < 64:
                delayed_epoch_after_cs = epoch_delay_early[epoch_after_cs]
            else:
                delayed_epoch_after_cs = epoch_after_cs

        min_scale_id = 0

        channel_choices = []
        for i in range(len(self.stage_out_channels)):
            for _ in range(self.stage_repeats[i]):
                if select_all_channels:
                    channel_choices = [len(self.candidate_scales) - 1] * sum(self.stage_repeats)
                else:
                    channel_scale_start = max(min_scale_id, len(self.candidate_scales) - delayed_epoch_after_cs - 2)
                    channel_choice = random.randint(channel_scale_start, len(self.candidate_scales) - 1)
                    # In sparse mode, channel_choices is the indices of candidate_scales
                    channel_choices.append(channel_choice)
        return channel_choices


if __name__ == '__main__':
    from src.factory.config_factory import _C as cfg
    import torch
    import numpy as np
    import random

    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    np.random.seed(42)
    random.seed(42)
    
    cfg.INPUT.RESIZE = (112, 112)
    cfg.DB.NUM_CLASSES = 7
    cfg.SPOS.USE_SE = True
    cfg.SPOS.LAST_CONV_AFTER_POOLING = True
    cfg.SPOS.CHANNELS_LAYOUT = "OneShot"
    graph = SPOS(cfg)

    for m in graph.model.modules():
        if hasattr(m, 'copy_weight'):
            m.copy_weight()

    # block_choice = [0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0]
    # channel_choice = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    # x = torch.ones(2,3,112,112)
    # out = graph.model(x, block_choice, channel_choice)
    # print(out)
