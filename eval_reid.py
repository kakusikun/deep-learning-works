import argparse
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
from tools.eval_reid_metrics import eval_single_query, eval_recall, evaluate
from model.OSNetv2 import osnet_x1_0
from config.config_manager import _C as cfg
from data.build_loader import build_reid_loader
from model.managers.manager_reid_trick import TrickManager
from model.managers.manager_reid_trick_att import AttentionManager
from tools.logger import setup_logger
from engine.engines.engine_reid_trick import ReIDEngine
import numpy as np
from tools.logger import setup_logger



parser = argparse.ArgumentParser(description="PyTorch Template MNIST Training")
parser.add_argument("--config", default="", help="path to config file", type=str)
parser.add_argument("--opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)
parser.add_argument("--type", default="cmc", help="evaluation type", type=str)
parser.add_argument("--manager", default="trick", help="evaluation type", type=str)
parser.add_argument("--cache", default="", help="evaluation type", type=str)

args = parser.parse_args()

if args.config != "":
    cfg.merge_from_file(args.config)
if args.opts != None:
    cfg.merge_from_list(args.opts)

log_name = "{}_evaluation_{}_{}".format(args.type, cfg.DATASET.NAME, cfg.EVALUATE.split("/")[-1])
logger = setup_logger("./evaluation/", log_name)
logger.info("Running with config:\n{}".format(cfg))

action = input("Config Confirmed ? (Y/N)").lower().strip()
if action == 'y':
    
    use_gpu = True
    metric = 'cosine'
    unitnorm_feat = True    

    if cfg.MODEL.PRETRAIN == "outside":
        core = osnet_x1_0(cfg.MODEL.NUM_CLASSES)
        checkpoint = torch.load(cfg.EVALUATE)
        model_state = core.state_dict()
        checkpointRefine = {}             
        for k, v in checkpoint.items():
            if k in model_state and torch.isnan(v).sum() == 0:
                checkpointRefine[k] = v
                # logger.info("{:60} ...... loaded".format(k))
            else:
                logger.info("{:60} ...... skipped".format(k))    
        model_state.update(checkpointRefine)
        core.load_state_dict(model_state)
    else:
        if args.manager == 'trick':
            model_manager = TrickManager(cfg)
            core = model_manager.model
        elif args.manager == 'attention':
            model_manager = AttentionManager(cfg)
            core = model_manager.model

    core = core.cuda()
    core.eval()

    _, qdata, gdata = build_reid_loader(cfg)

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
        logger.info("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

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
        logger.info("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    if metric == 'cosine':
        distmat =  1 - F.linear(qf, gf)
        distmat = distmat.numpy()
    elif metric == 'euclidean':
        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()

    if args.type == 'cmc':
        if args.cache:
            np.save("./{}_dismat.npy".format(args.cache), distmat)
            np.save("./{}_q_pids.npy".format(args.cache), q_pids)
            np.save("./{}_g_pids.npy".format(args.cache), g_pids)
            np.save("./{}_q_camids.npy".format(args.cache), q_camids)
            np.save("./{}_g_camids.npy".format(args.cache), g_camids)

        logger.info("Computing Single Query CMC")
        cmc = eval_single_query(distmat, q_pids, g_pids, q_camids, g_camids)

        logger.info("Results ----------")
        logger.info("CMC curve")
        for r in [1, 5, 10, 20]:
            logger.info("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        logger.info("------------------")

        logger.info("Computing Multiple Query CMC")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

        logger.info("Results ----------")
        logger.info("mAP: {:.4%}".format(mAP))
        logger.info("CMC curve")
        for r in [1, 5, 10, 20]:
            logger.info("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        logger.info("------------------")

    elif args.type == 'recall':
        eval_recall(distmat, q_pids, g_pids, q_camids, g_camids, save=True, name=log_name)
 
else:
    sys.exit(1)