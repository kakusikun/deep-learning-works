from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import os.path as osp
from PIL import Image
# import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy import array,argmin

import torch
import torch.nn as nn
from torch.backends import cudnn
from tools import bcolors
from tools.image import transform_preds

import logging
from tqdm import tqdm
logger = logging.getLogger("logger")

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def ctdet_post_process(dets, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret

def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()
    
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
      reg = _tranpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
      wh = wh.view(batch, K, cat, 2)
      clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
      wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
      wh = wh.view(batch, K, 2)
    
    valid_object = wh[:,:,0].gt(0) * wh[:,:,1].gt(0)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)[valid_object]
      
    return detections


def pdist(A, p, verbose=False):
    if p == 1:
        b = []
        if verbose:
            for i in tqdm(range(1, A.shape[0]), desc="Compute pdist"):
                b.append((A[i-1] - A[i:]).abs().sum(dim=1))
        else:
            for i in range(1, A.shape[0]):
                b.append((A[i-1] - A[i:]).abs().sum(dim=1))
        b = torch.cat(b)
    if p == 2:
        _pdist = lambda A, B: torch.sqrt(A.pow(2).sum(1, keepdim=True) - 2 * torch.mm(A, B.t()) + B.pow(2).sum(1, keepdim=True).t())
        dist = _pdist(A,A)
        mask = torch.triu(torch.ones_like(dist)) == 0
        b = dist[mask.t()]
    return b

def pair_idx_to_dist_idx(d, i, j):
    """
    :param d: numer of elements
    :param i: np.array. i < j in every element
    :param j: np.array
    :return:
    """
    assert np.sum(i < j) == len(i)
    index = d * i - i * (i + 1) / 2 + j - 1 - i
    return index.astype(int)


def dist_idx_to_pair_idx(d, i):
    """
    :param d: number of samples
    :param i: np.array
    :return:
    """
    if i.size() == 0:
        return None
    b = 1 - 2 * d
    x = torch.floor((-b - torch.sqrt(b ** 2 - 8 * i)) / 2).long()
    y = (i + (x * (b + x + 2) / 2).float() + 1).long()
    return x, y
    
def deploy_gpu(cfg):
    cudnn.benchmark = True
    gpu = ""
    for _gpu in cfg.MODEL.GPU:
        gpu += "{},".format(_gpu)
    gpu = gpu[:-1]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    logger.info("Using GPU: {}{}{}{}".format(bcolors.RESET, bcolors.OKGREEN, gpu, bcolors.RESET))

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
#     mkdir_if_missing(osp.dirname(fpath))
#     torch.save(state, fpath)
#     if is_best:
#         shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

# def _traceback(D):
#     i,j = array(D.shape)-1
#     p,q = [i],[j]
#     while (i>0) or (j>0):
#         tb = argmin((D[i,j-1], D[i-1,j]))
#         if tb == 0:
#             j -= 1
#         else: #(tb==1)
#             i -= 1
#         p.insert(0,i)
#         q.insert(0,j)
#     return array(p), array(q)

# def dtw(dist_mat):
#     m, n = dist_mat.shape[:2]
#     dist = np.zeros_like(dist_mat)
#     for i in range(m):
#         for j in range(n):
#             if (i == 0) and (j == 0):
#                 dist[i, j] = dist_mat[i, j]
#             elif (i == 0) and (j > 0):
#                 dist[i, j] = dist[i, j - 1] + dist_mat[i, j]
#             elif (i > 0) and (j == 0):
#                 dist[i, j] = dist[i - 1, j] + dist_mat[i, j]
#             else:
#                 dist[i, j] = \
#                     np.min(np.stack([dist[i - 1, j], dist[i, j - 1]], axis=0), axis=0) \
#                     + dist_mat[i, j]
#     path = _traceback(dist)
#     return dist[-1,-1]/sum(dist.shape), dist, path

# def read_image(img_path):
#     got_img = False
#     if not osp.exists(img_path):
#         raise IOError("{} does not exist".format(img_path))
#     while not got_img:
#         try:
#             img = Image.open(img_path).convert('RGB')
#             got_img = True
#         except IOError:
#             print("IOError incurred when reading '{}'. Will Redo. Don't worry. Just chill".format(img_path))
#             pass
#     return img

# def img_to_tensor(img,transform):
#     img = transform(img)
#     img = img.unsqueeze(0)
#     return img

# def show_feature(x):
#     for j in range(len(x)):
#         for i in range(len(64)):
#             ax = plt.subplot(4,16,i+1)
#             ax.set_title('No #{}'.format(i))
#             ax.axis('off')
#             plt.imshow(x[j].cpu().data.numpy()[0,i,:,:],cmap='jet')
#         plt.show()

# def feat_flatten(feat):
#     shp = feat.shape
#     feat = feat.reshape(shp[0] * shp[1], shp[2])
#     return feat

# def show_similar(local_img_path, img_path, similarity, bbox):
#     img1 = cv2.imread(local_img_path)
#     img2 = cv2.imread(img_path)
#     img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#     img1 = cv2.resize(img1, (64, 128))
#     img2 = cv2.resize(img2, (64, 128))
#     cv2.rectangle(img1, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 1)

#     p = np.where(similarity == np.max(similarity))
#     y, x = p[0][0], p[1][0]
#     cv2.rectangle(img2, (x - bbox[2] / 2, y - bbox[3] / 2), (x + bbox[2] / 2, y + bbox[3] / 2), (0, 255, 0), 1)
#     plt.subplot(1, 3, 1).set_title('patch')
#     plt.imshow(img1)
#     plt.subplot(1, 3, 2).set_title(('max similarity: ' + str(np.max(similarity))))
#     plt.imshow(img2)
#     plt.subplot(1, 3, 3).set_title('similarity')
#     plt.imshow(similarity)

# def show_alignedreid(local_img_path, img_path, dist):
#     def drow_line(img, similarity):
#         for i in range(1, len(similarity)):
#             cv2.line(img, (0, i*16), (63, i*16), color=(0,255,0))
#             cv2.line(img, (96, i*16), (160, i*16), color=(0,255,0))
#     def drow_path(img, path):
#         for i in range(len(path[0])):
#             cv2.line(img, (64, 8+16*path[0][i]), (96,8+16*path[1][i]), color=(255,255,0))
#     img1 = cv2.imread(local_img_path)
#     img2 = cv2.imread(img_path)
#     img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#     img1 = cv2.resize(img1, (64,128))
#     img2 = cv2.resize(img2, (64,128))
#     img = np.zeros((128,160,3)).astype(img1.dtype)
#     img[:,:64,:] = img1
#     img[:,-64:,:] = img2
#     drow_line(img, dist)
#     d,D,sp = dtw(dist)
#     origin_dist = np.mean(np.diag(dist))
#     drow_path(img, sp)
#     plt.subplot(1,2,1).set_title('Aligned distance: %.4f \n Original distance: %.4f' %(d,origin_dist))
#     plt.subplot(1,2,1).set_xlabel('Aligned Result')
#     plt.imshow(img)
#     plt.subplot(1,2,2).set_title('Distance Map')
#     plt.subplot(1,2,2).set_xlabel('Right Image')
#     plt.subplot(1,2,2).set_ylabel('Left Image')
#     plt.imshow(dist)
#     plt.subplots_adjust(bottom=0.1, left=0.075, right=0.85, top=0.9)
#     cax = plt.axes([0.9, 0.25, 0.025, 0.5])
#     plt.colorbar(cax = cax)
#     plt.show()

def merge_feature(feature_list, shp, sample_rate = None):
    def pre_process(torch_feature_map):
        numpy_feature_map = torch_feature_map.cpu().data.numpy()[0]
        numpy_feature_map = numpy_feature_map.transpose(1,2,0)
        shp = numpy_feature_map.shape[:2]
        return numpy_feature_map, shp
    def resize_as(tfm, shp):
        nfm, shp2 = pre_process(tfm)
        scale = shp[0]/shp2[0]
        nfm1 = nfm.repeat(scale, axis = 0).repeat(scale, axis=1)
        return nfm1
    final_nfm = resize_as(feature_list[0], shp)
    for i in range(1, len(feature_list)):
        temp_nfm = resize_as(feature_list[i],shp)
        final_nfm = np.concatenate((final_nfm, temp_nfm),axis =-1)
    if sample_rate > 0:
        final_nfm = final_nfm[0:-1:sample_rate, 0:-1,sample_rate, :]
    return final_nfm
