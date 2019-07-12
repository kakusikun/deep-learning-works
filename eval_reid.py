from tqdm import tqdm
import torch
import torch.nn.functional as F
from tools.eval_reid_metrics import evaluate
from model.OSNetv2 import osnet_x1_0
from config.config_manager import _C as cfg
from data.build_loader import build_reid_loader
import numpy as np

cfg.merge_from_file("./reid.yml")
cfg.EVALUATE = "/home/allen/Downloads/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"
use_gpu = True
metric = 'euclidean'
unitnorm_feat = False

_, qdata, gdata = build_reid_loader(cfg)

core = osnet_x1_0(cfg.MODEL.NUM_CLASSES)
checkpoint = torch.load(cfg.EVALUATE)
model_state = core.state_dict()
checkpointRefine = {}             
for k, v in checkpoint.items():
    if k in model_state and torch.isnan(v).sum() == 0:
        checkpointRefine[k] = v
        print("{:60} ...... loaded".format(k))
    else:
        print("{:60} ...... skipped".format(k))    
model_state.update(checkpointRefine)
core.load_state_dict(model_state)
core = core.cuda()
core.eval()

with torch.no_grad():   
    qf, q_pids, q_camids = [], [], []
    for batch in tqdm(qdata, desc="Validation"):
        
        imgs, pids, camids = batch
        if use_gpu: imgs = imgs.cuda()
        
        features = core(imgs)

        if unitnorm_feat:
            features = F.normalize(features)
        
        qf.append(features.cpu())
        q_pids.extend(pids)
        q_camids.extend(camids)

    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

    gf, g_pids, g_camids = [], [], []
    for batch in tqdm(gdata, desc="Validation"):
        
        imgs, pids, camids = batch
        if use_gpu: imgs = imgs.cuda()
        
        features = core(imgs)

        if unitnorm_feat:
            features = F.normalize(features)
        
        gf.append(features.cpu())
        g_pids.extend(pids)
        g_camids.extend(camids)

    gf = torch.cat(gf, 0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

if metric == 'cosine':
    distmat =  1 - F.linear(qf, gf)
    distmat = distmat.numpy()
elif metric == 'euclidean':
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

print("Computing CMC and mAP")
cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

print("Results ----------")
print("mAP: {:.1%}".format(mAP))
print("CMC curve")
for r in [1, 5, 10, 20]:
    print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
print("------------------")