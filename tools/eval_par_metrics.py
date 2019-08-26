from __future__ import print_function, absolute_import
import numpy as np
import copy
from collections import defaultdict
import sys
from tqdm import tqdm

def eval_par_accuracy(predict, gt):
    known_gt = gt > -1

    TPR = []
    FPR = []
    total_precision = []
    
    for thresh in tqdm(range(0, 100), desc="PAR Accuracy"):
        thresh *= 0.01
        predicted_gt = np.zeros_like(predict)
        predicted_gt[predict >= thresh] = 1

        total_precision.append((predicted_gt[known_gt] == gt[known_gt]).astype(float).mean())

        TPR.append((predicted_gt[known_gt] * gt[known_gt]).sum() / gt[known_gt].sum())

        inverse_gt = np.invert(gt[known_gt].astype(bool)).astype(float)
        FPR.append((predicted_gt[known_gt] * inverse_gt).sum() / inverse_gt.sum())

    return TPR, FPR, total_precision