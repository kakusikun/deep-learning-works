import os
import sys
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import pdist
from collections import OrderedDict
from manager.OSNet_iabn import osnet_x1_0
from manager.utility import JointLoss, MultilabelLoss, DiscriminativeLoss
from manager.base_manager import BaseManager
import logging
from tqdm import tqdm
logger = logging.getLogger("logger")

class MARManager(BaseManager):
    def __init__(self, cfg):
        super(MARManager, self).__init__(cfg)        

        self.s = 30.0   
        self.p = 0.005    
        self.num_sample = 0
        self.l1 = 0.0002
        self.l2 = 50
        self.b = 0.2

        if cfg.TASK == "reid":
            self._make_model()
            self._make_loss()
        else:
            logger.info("Task {} is not supported".format(cfg.TASK))  
            sys.exit(1)

        self._check_model()    
                        
    def _make_model(self):
        self.model = Model(self.cfg)

    def _make_loss(self):
        self.AL_loss = nn.CrossEntropyLoss()
        self.RJ_loss = JointLoss()
        self.CML_loss = MultilabelLoss(batch_size=self.cfg.INPUT.TRAIN_BS)
        self.MDL_loss = DiscriminativeLoss(mining_ratio=self.p)

        self.loss_has_param = []
        self.loss_name = ["AL", "RJ", "CML", "MDL"]

        def loss_func(src_feat, trt_feat, src_sim, trt_sim, src_target, trt_cams, trt_idx, epoch):
            agents = F.normalize(self.model.module.id_fc.weight, dim=1).detach()
            multilabels = F.softmax(self.s * trt_feat.mm(agents.t()), dim=1)
            each_loss = [self.AL_loss(self.s * src_sim, src_target),
                         self.RJ_loss(src_feat, agents, src_target, src_sim.detach(), trt_feat, trt_sim.detach()),
                         self.CML_loss(torch.log(multilabels), trt_cams)]
            if epoch < 0:
                each_loss.append(torch.Tensor([0]).cuda())
            else:
                multilabels = multilabels.detach()
                is_init_mask = self.is_initialized[trt_idx]
                init_trt_idx = trt_idx[is_init_mask]
                uninit_trt_idx = trt_idx[~is_init_mask]
                self.multilabel_buffer[uninit_trt_idx] = multilabels[~is_init_mask]
                self.is_initialized[uninit_trt_idx] = 1
                self.multilabel_buffer[init_trt_idx] = 0.9 * self.multilabel_buffer[init_trt_idx] + 0.1 * multilabels[is_init_mask]
                each_loss.append(self.MDL_loss(trt_feat, self.multilabel_buffer[trt_idx]))
                
            loss = each_loss[3] + self.l1 * each_loss[2] + self.l2 * (each_loss[0] + self.b * each_loss[1])
            return loss.squeeze(), each_loss

        self.loss_func = loss_func
    
    def get_similarity_cam(self, data):
        logger.info('Extract features')
        tmp_path = self.cfg.OUTPUT_DIR.split(os.path.basename(self.cfg.OUTPUT_DIR))[0]
        if os.path.exists(os.path.join(tmp_path, "Similarity.pt")):
            logger.info('Cache is found at {}'.format(tmp_path))
            sims = torch.load(os.path.join(tmp_path, "Similarity.pt")) 
            cams = torch.load(os.path.join(tmp_path, "cams.pt"))
        else:
            self.model.eval()
            sims = []
            cams = []
            with torch.no_grad():   
                for batch in tqdm(data, desc="Extract"):                
                    imgs, _, camid, = batch
                    imgs = imgs.cuda()
                    
                    _, sim = self.model(imgs, extract=True)
                    sims.append(sim)
                    cams.append(camid)

            sims = torch.cat(sims, dim=0)
            cams = torch.cat(cams, dim=0)
            torch.save(sims, os.path.join(tmp_path, "Similarity.pt")) 
            torch.save(cams, os.path.join(tmp_path, "cams.pt"))
        logger.info("Similarity matrix: {}x{}".format(sims.shape[0], sims.shape[1]))
        return sims, cams

    def stats_initialization(self, trt_loader):
        sims, cams = self.get_similarity_cam(trt_loader)
        multilabels = F.softmax(self.s * sims.detach(), dim=1)
        self.multilabel_buffer = torch.zeros_like(multilabels).cuda()
        self.is_initialized = self.multilabel_buffer.sum(dim=1) != 0 
        # eq(2), agreement
        tmp_path = self.cfg.OUTPUT_DIR.split(os.path.basename(self.cfg.OUTPUT_DIR))[0]
        if os.path.exists(os.path.join(tmp_path, "pa.pt")):
            logger.info('Cache is found at {}'.format(tmp_path))
            pairwise_agreements = torch.load(os.path.join(tmp_path, "pa.pt")) 
        else:
            pairwise_agreements = 1 - pdist(multilabels, p=1, verbose=True)/2
            torch.save(pairwise_agreements, os.path.join(tmp_path, "pa.pt"))
        log_multilabels = torch.log(multilabels)
        self.CML_loss.init_centers(log_multilabels, cams)
        self.MDL_loss.init_threshold(pairwise_agreements.cpu().numpy())

    

def weights_init_kaiming(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(module):
    for m in module.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
            
class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        if cfg.MODEL.NAME == 'osnet':
            self.in_planes = 512
            if cfg.MODEL.PRETRAIN == "outside":
                self.backbone = osnet_x1_0(task='trick') 
            else:
                self.backbone = osnet_x1_0(cfg.DB.NUM_CLASSES, task='trick')        
        else:
            logger.info("{} is not supported".format(cfg.MODEL.NAME))

        self.gap = nn.AdaptiveAvgPool2d(1)        
        self.BNNeck = nn.BatchNorm1d(self.in_planes)
        self.BNNeck.bias.requires_grad_(False)  # no shift
        self.BNNeck.apply(weights_init_kaiming)

        self.num_classes = cfg.DB.NUM_CLASSES
        self.id_fc = nn.Linear(self.in_planes, self.num_classes, bias=False)        
        self.id_fc.apply(weights_init_classifier)
    
    def forward(self, x, extract=False):
        # use trick: BNNeck, feature before BNNeck to triplet GAP and feature w/o fc forward in backbone
        x = self.backbone(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        feat = self.BNNeck(x)
        feat = F.normalize(feat, dim=1)
        if not self.training:
            if extract:
                w = F.normalize(self.id_fc.weight, dim=1)        
                sim = feat.mm(w.t()) 
                return feat, sim
            return feat
        w = F.normalize(self.id_fc.weight, dim=1)        
        sim = feat.mm(w.t()) 
        return feat, sim
