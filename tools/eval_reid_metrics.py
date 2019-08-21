from __future__ import print_function, absolute_import
import numpy as np
import copy
from collections import defaultdict
import sys
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, N=100):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed N times (default: N=100).
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep] 
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc, AP = 0., 0.
        for repeat_idx in range(N):
            mask = np.zeros(len(orig_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_orig_cmc = orig_cmc[mask]
            _cmc = masked_orig_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)
            # compute AP
            num_rel = masked_orig_cmc.sum()
            tmp_cmc = masked_orig_cmc.cumsum()
            tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * masked_orig_cmc
            AP += tmp_cmc.sum() / num_rel
        cmc /= N
        AP /= N
        all_cmc.append(cmc)
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query

    for q_idx in tqdm(range(num_q), desc="CMC"):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches

        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def eval_single_query(distmat, q_pids, g_pids, q_camids, g_camids):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    num_valid_q = 0. # number of valid query
    num_success_q = np.zeros(50)
    for q_idx in tqdm(range(num_q), desc="CMC"):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        num_valid_q += sum(matches[q_idx][keep])

        num_match = 0
        for idx, match in enumerate(matches[q_idx][keep]):
            if match == 0:
                num_match += 1
                for i in range(50):
                    if num_match == i+1:
                        num_success_q[i] += sum(matches[q_idx][keep][:idx])
                if num_match == 50:    
                    break

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    cmc = num_success_q / num_valid_q

    return cmc   


def eval_recall(distmat, q_pids, g_pids, q_camids, g_camids, save=False, name="recall"):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    indices = np.tile(g_pids, (q_pids.shape[0],1))
    matches = (indices == q_pids[:, np.newaxis]).astype(np.int32)
    confmat = 1 - distmat

    # compute cmc curve for each query
    recalls = []
    precisions = []
    for thresh in tqdm(range(0,101,1), desc="Recall"):
        recall = []
        precision = []
        thresh *= 0.01
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query        
            remove = (g_pids == q_pid) & (g_camids == q_camid)
            keep = np.invert(remove)

            # compute number of true positive and positive given query
            TP = np.sum(confmat[q_idx][keep][matches[q_idx][keep]==1] >= thresh, dtype=np.float32)
            PP = np.sum(confmat[q_idx][keep] >= thresh, dtype=np.float32)
            P = np.sum(matches[q_idx][keep], dtype=np.float32)
            recall.append(TP/P)
            precision.append(TP/PP if PP != 0.0 else 0.0)

        recall = np.array(recall)
        recalls.append(np.mean(recall))
        precision = np.array(precision)
        precisions.append(np.mean(precision))

    thresh = np.array(list(range(0,101,1)))*0.01
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    df = pd.DataFrame({'thresh': thresh, 'recall': recalls, 'precision': precisions})    
    df.to_csv("./evaluation/{}.csv".format(name), index=False)

    if save:                
        plt.figure(figsize=(12, 5))
        sns.set()
        sns.lineplot(x="thresh", y="recall", data=df, color="b", label="recall")
        sns.lineplot(x="thresh", y="precision", data=df, color="r", label="precision")
        for i in range(0,101,10):
            i *= 0.01
            offset = -0.1 if i <= 0.5 else 0.1
            r_value = df[df['thresh']==i]['recall'].iloc[0]
            plt.text(x=i, y=r_value+offset, s="{:.3f}".format(r_value), color="b")
            plt.scatter(i, r_value, s=50, c="b")
            p_value = df[df['thresh']==i]['precision'].iloc[0]
            plt.text(x=i, y=r_value+2*offset, s="{:.3f}".format(p_value), color="r")
            plt.scatter(i, p_value, s=50, c="r")
        plt.xlabel("Threshold")
        plt.ylabel("Value")
        plt.legend(bbox_to_anchor=(0.85, 0.95), loc='upper left', borderaxespad=0.5)
        plt.title("{} => {}".format(name.split("_")[3], name.split("_")[1]))
        plt.savefig("{}.jpg".format(name))
        

def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, use_metric_cuhk03=False):
    if use_metric_cuhk03:
        return eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    else:
        return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)

if __name__ == '__main__':
    distmat = np.load("./dismat.npy")
    q_pids = np.load("./q_pids.npy")
    g_pids = np.load("./g_pids.npy")
    q_camids = np.load("./q_camids.npy")
    g_camids = np.load("./g_camids.npy")
    evaluate(distmat, q_pids, g_pids, q_camids, g_camids)