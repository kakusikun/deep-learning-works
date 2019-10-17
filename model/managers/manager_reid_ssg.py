import os
import sys
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from model.OSNetv2 import osnet_x1_0
from model.ResNet import ResNet, BasicBlock
from model.utility import TripletLoss, ClusterAssignment
from model.manager import TrainingManager
import logging
logger = logging.getLogger("logger")

class SSGManager(TrainingManager):
    def __init__(self, cfg):
        super(SSGManager, self).__init__(cfg)        

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
        if self.cfg.MODEL.NAME == 'osnet':
            feat_dim = 512   

        triplet_loss = TripletLoss()
        KD_loss = nn.KLDivLoss(size_average=False)
        self.loss_has_param = []
        self.loss_name = ["triplet_global", "triplet_whole", "triplet_upper", "triplet_lower", "KD"]

        def loss_func(feats, target):
            each_loss = [triplet_loss(feats[1], target[0])[0]]

            for i, feat in enumerate(feats[0]):
                each_loss.append(triplet_loss(feat, target[i])[0])

            kd_loss = 0.0
            for feat in feats[2]:
                feat_p = target_distribution(feat)
                kd_loss += KD_loss(feat.log(), feat_p) / feat.shape[0]
            each_loss.append(kd_loss)

            loss = 0.0
            for _loss in each_loss:
                loss += _loss            

            return loss, each_loss

        self.loss_func = loss_func
    
    def extract_features(self, data, cycle):
        self.model.eval()
        _feats = defaultdict(list)
        with torch.no_grad():   
            for batch in tqdm(data, desc="Extract {}".format(cycle)):                
                imgs, _, _, = batch
                if use_gpu: imgs = imgs.cuda()
                
                feat = self.model(imgs)[0]
                hflip_feat = self.model(fliplr(imgs))[0]
                for i in range(len(feat)):
                    feat[i] += hflip_feat[i]
                    feat[i] = F.normalize(feat[i])
                    _feats[i].append(feat[i])
        feats = []
        for i in _feats.keys():
            feats.append(torch.cat(_feats[i]))        
        return feats
    
    def get_self_label(self, dists, cycle):
        labels_list = []
        for i in range(len(dists)):
            # if cycle==0:                
            ####DBSCAN cluster
            tri_mat = np.triu(dists[i],1)       # tri_mat.dim=2
            tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
            tri_mat = np.sort(tri_mat,axis=None)
            top_num = np.round(1.6e-3*tri_mat.size).astype(int)
            eps = tri_mat[:top_num].mean()
            logger.info('eps in cluster: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps,min_samples=4, metric='precomputed', n_jobs=8)
            # else:
            #     cluster = self.cluster_handles[i]
            #### select & cluster images as training set of this epochs
            labels = cluster.fit_predict(dists[i])            
            num_ids = len(set(labels)) - 1  ##for DBSCAN cluster
            logger.info('At {}th cycle, {}th feature has {} training ids'.format(cycle, i, num_ids))
            labels_list.append(labels)

        return labels_list

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def target_distribution(batch):
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch ** 2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

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
                self.backbone = osnet_x1_0(cfg.MODEL.NUM_CLASSES, task='trick')        
        else:
            logger.info("{} is not supported".format(cfg.MODEL.NAME))

        self.GAP = nn.AdaptiveAvgPool2d(1)        
        # self.BNNeck = nn.BatchNorm1d(self.in_planes)
        # self.BNNeck.bias.requires_grad_(False)  # no shift
        # self.BNNeck.apply(weights_init_kaiming)

        self.fc = nn.Linear(self.in_planes, self.in_planes, bias=False)     
        self.fc_bn = nn.BatchNorm1d(self.in_planes)
        self.fc_relu = nn.ReLU(inplace=True)
        self.fc.apply(weights_init_classifier)
        self.fc_bn.apply(weights_init_kaiming)

        self.assignment = ClusterAssignment(cluster_number=cfg.INPUT.SIZE_TRAIN // cfg.REID_SIZE_PERSON, embedding_dimension=512)
    
    def forward(self, x):
        feat = self.backbone(x) 

        h = feat.size(2) 
        x1 = [feat]                   
        x1.extend[feat[:, :, h // 2 * s: h // 2 * (s+1), :] for s in range(2)]
        for i, x1x in enumerate(x1):
            x1x = self.GAP(x1xx)
            x1[i] = (x1x.view(x1x.size(0), -1)) 
        x2 = self.fc_relu(self.fc_bn(self.fc(x1[0])))

        if not self.training:
            x1 = torch.cat(x1, dim=1)
            return x1, x2  

        x3 = self.assignment(torch.cat(x1, dim=1)) 
        return x1, x2, x3   
