import os
import sys
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist
from collections import OrderedDict
from model.OSNet_iabn import osnet_x1_0
from model.utility import JointLoss, MultilabelLoss, DiscriminativeLoss
from model.manager import TrainingManager
import logging
logger = logging.getLogger("logger")

class MARManager(TrainingManager):
    def __init__(self, cfg):
        super(MARManager, self).__init__(cfg)        

        if cfg.TASK == "reid":
            self._make_model()
            self._make_loss()
        else:
            logger.info("Task {} is not supported".format(cfg.TASK))  
            sys.exit(1)

        self._check_model()    
        self.s = 30.0   
        self.p = 0.005    
        self.multilabel_buffer = None
        self.l1 = 0.0002
        self.l2 = 50
        self.b = 0.2
                        
    def _make_model(self):
        self.model = Model(self.cfg)

    def _make_loss(self):
        self.AL_loss = nn.CrossEntropyLoss()
        self.RJ_loss = JointLoss()
        self.CML_loss = MultilabelLoss()
        self.MDL_loss = DiscriminativeLoss(mining_ratio=self.p)

        self.loss_has_param = []
        self.loss_name = ["AL", "RJ", "CML", "MDL"]

        def loss_func(src_feat, trt_feat, src_sim, trt_sim, src_target, trt_cams, trt_idx):
            agents = F.normalize(self.model.id_fc.weight, dim=1).detach()
            multilabels = F.softmax(self.s * trt_feat.mm(agents.t()), dim=1)
            self.multilabel_buffer[trt_idx] = 0.9 * self.multilabel_buffer[trt_idx] + 0.1 * multilabels.detach()
            each_loss = [self.AL_loss(self.s * src_sim, src_target),
                         self.RJ_loss(src_feat, agents, src_target, src_sim.detach(), trt_feat, trt_sim.detach()),
                         self.CML_loss(torch.log(multilabels), trt_cams),
                         self.MDL_loss(trt_feat, self.multilabel_buffer[trt_idx])]
            loss = each_loss[3] + self.l1 * each_loss[2] + self.l2 * (each_loss[0] + self.b * each_loss[1])
            return loss, each_loss

        self.loss_func = loss_func
    
    def get_similarity_cam(self, data):
        logger.info('Extract features')
        self.model.eval()
        sims = []
        cams = []
        with torch.no_grad():   
            for batch in tqdm(data, desc="Extract"):                
                imgs, _, camid, = batch
                imgs = imgs.cuda()
                
                _, sim = self.model(imgs)
                sims.append(sim)
                cams.append(cams)

        sims = torch.cat(sims, dim=0)
        cams = torch.cat(cams, dim=0)
        logger.info("Similarity matrix: {}x{}".format(sims.shape[0], sims.shape[1]))
        return sims, cams

    def stats_initialization(self, trt_loader):
        sims, cams = self.get_similarity_cam(trt_loader)
        multilabels = F.softmax(self.s * sims, dim=1)
        self.multilabel_buffer = multilabels.detach()
        ml_np = multilabels.cpu().numpy()
        # eq(2), agreement
        pairwise_agreements = 1 - pdist(ml_np, 'minkowski', p=1)/2
        log_multilabels = torch.log(multilabels)
        self.CML_loss.init_centers(log_multilabels, cams)
        self.MDL_loss.init_threshold(pairwise_agreements)

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
        if cfg.MODEL.NAME == 'resnet18':
            self.in_planes = 512
            self.backbone = ResNet(last_stride=1, block=BasicBlock, layers=[2,2,2,2])
        elif cfg.MODEL.NAME == 'rmnet':
            self.in_planes = 256
            self.backbone = RMNet(b=[4,8,10,11], cifar10=False, reid=True, trick=True)
        elif cfg.MODEL.NAME == 'osnet':
            self.in_planes = 512
            if cfg.MODEL.PRETRAIN == "outside":
                self.backbone = osnet_x1_0(task='trick') 
            else:
                self.backbone = osnet_x1_0(cfg.MODEL.NUM_CLASSES, task='trick')        
        else:
            logger.info("{} is not supported".format(cfg.MODEL.NAME))

        self.gap = nn.AdaptiveAvgPool2d(1)        
        self.BNNeck = nn.BatchNorm1d(self.in_planes)
        self.BNNeck.bias.requires_grad_(False)  # no shift
        self.BNNeck.apply(weights_init_kaiming)

        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.id_fc = nn.Linear(self.in_planes, self.num_classes, bias=False)        
        self.id_fc.apply(weights_init_classifier)
    
    def forward(self, x):
        # use trick: BNNeck, feature before BNNeck to triplet GAP and feature w/o fc forward in backbone
        x = self.backbone(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        feat = self.BNNeck(x)
        feat = F.normalize(feat, dim=1)
        if not self.training:
            return feat      
        w = F.normalize(self.id_fc.weight, dim=1)        
        sim = feat.mm(w.t()) 

        return feat, sim
