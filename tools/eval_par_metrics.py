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

    attr_TPR = []
    attr_FPR = []
    attr_total_precision = []
    
    
    for thresh in tqdm(range(0, 100), desc="PAR Accuracy"):
        thresh *= 0.01
        predicted_gt = np.zeros_like(predict)
        predicted_gt[predict >= thresh] = 1

        attr_predicted_count = []
        attr_TP_count = []
        attr_FP_count = []
        attr_P_count = []
        attr_N_count = []
        for i in range(known_gt.shape[1]):
            attr_predicted_count.append((predicted_gt[:,i][known_gt[:,i]] == gt[:,i][known_gt[:,i]]).astype(float).sum())

            attr_TP_count.append((predicted_gt[:,i][known_gt[:,i]] * gt[:,i][known_gt[:,i]]).sum())
            attr_P_count.append(gt[:,i][known_gt[:,i]].sum())

            inverse_gt = np.invert(gt[:,i][known_gt[:,i]].astype(bool)).astype(float)
            attr_FP_count.append((predicted_gt[:,i][known_gt[:,i]] * inverse_gt).sum())
            attr_N_count.append(inverse_gt.sum())

        total_precision.append(np.array(attr_predicted_count).sum() / (known_gt.sum() + 1e-10))
        TPR.append(np.array(attr_TP_count).sum() / (np.array(attr_P_count).sum() + 1e-10))
        FPR.append(np.array(attr_FP_count).sum() / (np.array(attr_N_count).sum() + 1e-10))

        attr_total_precision.append(np.array(attr_predicted_count) / (known_gt.sum(axis=0) + 1e-10))
        attr_TPR.append(np.array(attr_TP_count) / (np.array(attr_P_count) + 1e-10))
        attr_FPR.append(np.array(attr_FP_count) / (np.array(attr_N_count) + 1e-10))

    return TPR, FPR, total_precision, attr_TPR, attr_FPR, attr_total_precision
