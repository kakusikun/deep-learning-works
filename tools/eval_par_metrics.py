from __future__ import print_function, absolute_import
import numpy as np
import copy
from collections import defaultdict
import sys
from sklearn.metrics import f1_score, recall_score, precision_score
from tqdm import tqdm

def eval_par_accuracy(predict_proba, gt):
    known_gt = gt > -1 

    attr_precs = []
    attr_recalls = []  
    
    for thresh in tqdm(range(0, 100), desc="PAR Accuracy"):
        thresh *= 0.01
        predict = np.zeros_like(predict_proba)
        predict[predict_proba >= thresh] = 1
        attr_prec = []
        attr_recall = [] 
        for i in range(known_gt.shape[1]):
            p = precision_score(gt[:,i][known_gt[:,i]], predict[:,i][known_gt[:,i]], average='macro')
            r = recall_score(gt[:,i][known_gt[:,i]], predict[:,i][known_gt[:,i]], average='macro')    
            attr_prec.append(p)
            attr_recall.append(r)

        attr_precs.append(attr_prec)
        attr_recalls.append(attr_recall)

    attr_precs = np.array(attr_precs)
    attr_recalls = np.array(attr_recalls)

    return attr_precs, attr_recalls
