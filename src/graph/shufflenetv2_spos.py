import os
import json
from sys import maxsize
from src.graph import *
from src.model.backbone.shufflenet_oneshot import ShuffleNetOneShot
from src.model.module.loss_module import CrossEntropyLossLS
import random

class _Model(nn.Module):
    def __init__(self, cfg, strides, stage_repeats, stage_out_channels):
        super().__init__()
        self.backbone = ShuffleNetOneShot(
            strides=strides,
            stage_repeats=stage_repeats,
            stage_out_channels=stage_out_channels,
            mode='v2'
        )
        self.head = ClassifierHead(cfg.MODEL.FEATSIZE, cfg.DB.NUM_CLASSES)
    def forward(self, x, block_choices, channel_choices):
        x = self.backbone(x, block_choices, channel_choices)
        x = self.head(x)
        return x

class ShuffleNetv2SPOS(BaseGraph):
    def __init__(self, cfg):
        self.strides = [1, 1, 2, 2, 2]
        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [24, 116, 232, 464]
        self.channel_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        super().__init__(cfg)

    def build(self):
        self.model = _Model(self.cfg, self.strides, self.stage_repeats, self.stage_out_channels)
    
        self.crit = {}
        self.crit['cels'] = CrossEntropyLossLS(self.cfg.DB.NUM_CLASSES)

        def loss_head(feat, batch):
            losses = {'cels':self.crit['cels'](feat, batch['target'])}
            loss = losses['cels']
            return loss, losses

        self.loss_head = loss_head
        self.lookup_table = self.get_lookup_table()

    def generate_block_candidates(self, epoch_after_search=None):
        block_candidates = []
        for num_repeats in self.stage_repeats:
            for i in range(num_repeats):
                if i == 0:
                    block_candidates.append([1,2])
                else:
                    if epoch_after_search >= 0 or epoch_after_search is None:
                        block_candidates.append([0, 1, 2])
                    else:
                        block_candidates.append([1, 2])
        return block_candidates

    def generate_channel_candidates(self, epoch_after_search=None):
        choice = list(range(len(self.channel_scales)))
        channel_candidates = []
        for num_repeats in self.stage_repeats:
            for _ in range(num_repeats):
                if epoch_after_search is None:                   
                    channel_candidates.append(choice)
                else:
                    if epoch_after_search >= 0:
                        channel_candidates.append(choice[(-1*(epoch_after_search//self.cfg.SPOS.CANDIDATE_RELAX_EPOCHS+2)):])
                    else:
                        channel_candidates.append(choice[-1:])
        return channel_candidates

    def get_lookup_table(self):
        root = os.getcwd()
        file_path = os.path.join(root, 'external/OneShot_flops.json')   

        if not os.path.exists(file_path):
            logger.info("FLOPs Table is not found, generating ...")
            dummy_input = torch.rand(1, 3, *self.cfg.INPUT.SIZE)
            lookup_table = self.model.backbone._get_lookup_table(dummy_input)
            with open(file_path, 'w') as f:
                json.dump(lookup_table, f)
            logger.info(f"Done. FLOPs Table is placed at {file_path}")

        with open(file_path, 'r') as f:
            lookup_table = json.load(f)
        
        return lookup_table

    def random_block_choices(self, epoch_after_search=None):
        block_candidates = self.generate_block_candidates(epoch_after_search)
        block_choices = []
        for i in range(sum(self.stage_repeats)):
            block_choices.append(random.choice(block_candidates[i]))
        return block_choices

    def random_channel_choices(self, epoch_after_search=None):
        """
        candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        mode: str, "dense" or "sparse". Sparse mode select # channel from candidate scales. Dense mode selects
              # channels between randint(min_channel, max_channel).
        """
        assert len(self.stage_repeats) == len(self.stage_out_channels[1:])
        # From [1.0, 1.2, 1.4, 1.6, 1.8, 2.0] to [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], warm-up stages are
        # not just 1 epoch, but 2, 3, 4, 5 accordingly.
        channel_candidates = self.generate_channel_candidates(epoch_after_search)

        #TODO: remove older method
        # epoch_delay_early = {0: 0,  # 8
        #                      1: 1, 2: 1,  # 7
        #                      3: 2, 4: 2, 5: 2,  # 6
        #                      6: 3, 7: 3, 8: 3, 9: 3,  # 5
        #                      10: 4, 11: 4, 12: 4, 13: 4, 14: 4,
        #                      15: 5, 16: 5, 17: 5, 18: 5, 19: 5, 20: 5,
        #                      21: 6, 22: 6, 23: 6, 24: 6, 25: 6, 27: 6, 28: 6,
        #                      29: 6, 30: 6, 31: 6, 32: 6, 33: 6, 34: 6, 35: 6, 36: 7,
        #                    }
        # epoch_delay_late = {0: 0,
        #                     1: 1,
        #                     2: 2,
        #                     3: 3,
        #                     4: 4, 5: 4,  # warm up epoch: 2 [1.0, 1.2, ... 1.8, 2.0]
        #                     6: 5, 7: 5, 8: 5,  # warm up epoch: 3 ...
        #                     9: 6, 10: 6, 11: 6, 12: 6,  # warm up epoch: 4 ...
        #                     13: 7, 14: 7, 15: 7, 16: 7, 17: 7,  # warm up epoch: 5 [0.4, 0.6, ... 1.8, 2.0]
        #                     18: 8, 19: 8, 20: 8, 21: 8, 22: 8, 23: 8}  # warm up epoch: 6, after 17, use all scales
        # select_all_channels = False
        # if epoch_after_search < 0:
        #     select_all_channels = True
        # else:
        #     if 0 <= epoch_after_search <= 23 and self.model.backbone.stage_out_channels[1] >= 64:
        #         delayed_epoch_after_cs = epoch_delay_late[epoch_after_search]
        #     elif 0 <= epoch_after_search <= 36 and self.model.backbone.stage_out_channels[1] < 64:
        #         delayed_epoch_after_cs = epoch_delay_early[epoch_after_search]
        #     else:
        #         delayed_epoch_after_cs = epoch_after_search

        # min_scale_id = 0

        # channel_choices = []
        # for i in range(len(self.model.backbone.stage_out_channels[1:])):
        #     for _ in range(self.model.backbone.stage_repeats[i]):
        #         if select_all_channels:
        #             channel_choices = [len(self.model.backbone.channel_scales) - 1] * sum(self.model.backbone.stage_repeats)
        #         else:
        #             channel_scale_start = max(min_scale_id, len(self.model.backbone.channel_scales) - delayed_epoch_after_cs - 2)
        #             channel_choice = random.randint(channel_scale_start, len(self.model.backbone.channel_scales) - 1)
        #             # In sparse mode, channel_choices is the indices of candidate_scales
        #             channel_choices.append(channel_choice)

        channel_choices = []
        for i in range(sum(self.stage_repeats)):
            channel_choices.append(random.choice(channel_candidates[i]))

        return channel_choices


if __name__ == '__main__':
    from src.factory.config_factory import _C as cfg
    import torch
    import numpy as np
    import random
    from copy import deepcopy

    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    np.random.seed(42)
    random.seed(42)
    
    cfg.INPUT.SIZE = (64, 64)
    cfg.DB.NUM_CLASSES = 200
    cfg.MODEL.FEATSIZE = 464
    graph = ShuffleNetv2SPOS(cfg)

    # b = graph.random_block_choices(-1)
    # b = graph.random_block_choices(10)
    # c = graph.random_channel_choices(-1)
    # c = graph.random_channel_choices(10)
    # c = graph.random_channel_choices(20)
    # c = graph.random_channel_choices(30)
    # c = graph.random_channel_choices(40)
    # c = graph.random_channel_choices(50)
    # c = graph.random_channel_choices(60)
    # c = graph.random_channel_choices(70)
    # c = graph.random_channel_choices(80)
    # c = graph.random_channel_choices(90)
    # c = graph.random_channel_choices(100)
    # c = graph.random_channel_choices(110)

    for m in graph.model.modules():
        if hasattr(m, 'copy_weight'):
            before_p = deepcopy(next(iter(m.parameters())))
            m.copy_weight()
            after_p = next(iter(m.parameters()))
            assert (after_p == before_p).sum() == 0

    # block_choice = [0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0]
    # channel_choice = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    # x = torch.ones(2,3,112,112)
    # out = graph.model(x, block_choice, channel_choice)
    # print(out)
