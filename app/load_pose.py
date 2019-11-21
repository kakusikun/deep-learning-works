import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
trans = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

import numpy as np
from scipy.spatial.distance import cdist
from scipy import stats

import cv2
from PIL import Image
from skimage.segmentation import slic
from skimage import io
from skimage import img_as_float

import os
import os.path as osp
from itertools import product
from app.pose.rtpose_vgg import get_model
from app.pose.post import decode_pose, plot_pose
from app.pose.preprocessing import rtpose_preprocess

import pickle
from colormath.color_diff_matrix import delta_e_cie2000
root = osp.dirname(osp.realpath(__file__))
infile = open(osp.join(root, 'lab-matrix.pk'),'rb')
lab_matrix = pickle.load(infile, encoding='latin1')
infile.close()
infile = open(osp.join(root, 'lab-colors.pk'),'rb')
lab_colors = pickle.load(infile, encoding='latin1')
infile.close()
lab_colors = np.array(lab_colors)


right_shoulder = 2
left_shoulder = 5
right_hip = 8
left_hip = 11
right_knee = 9
left_knee = 12
UPPER = [[2, 11], [5, 8]]
THIGH = [[8, 9], [11, 12]]
IMPORTANT = [2,5,8,9,11,12]

def load_pose_model(path):
    model = get_model('vgg19') 
    weight_name = path
    checkpoint = torch.load(weight_name)
    model_dict = model.state_dict()
    ckpt = {k.replace("module.", ""): v for k, v in checkpoint.items() if k.replace("module.", "") in model_dict and torch.isnan(v).sum() == 0}
    model_dict.update(ckpt)
    model.load_state_dict(model_dict)
    model = model.cuda()
    # model.float()
    model.eval()
    return model

class COLORS():
    BGR = np.array([[int(b),int(g),int(r)] for b,g,r in product([0,255],repeat = 3)])

    H = np.array([[  10000.],
                    [  0.],
                    [120.],
                    [ 60.],
                    [240.],
                    [ 44.],
                    [180.],
                    [  10000.],
                    [  10000.]])
    
    def __init__(self):
        self.color_names = ['Black',
                            'Red',
                            'Green',
                            'Yellow',
                            'Blue',
                            'Magenta',
                            'Aqua',
                            'Gray',
                            'White']        
        self.color_np = {'Black':(0,0,0),
                            'Red':(0,0,255),
                            'Green':(0,255,0),
                            'Yellow':(0,255,255),
                            'Blue':(255,0,0),
                            'Magenta':(255,0,255),
                            'Aqua':(255,255,0),
                            'Gray':(127,127,127),
                            'White':(255,255,255)}
    def get_color(self, c):
        return self.color_names[c]
    def get_color_value(self, c):
        return self.color_np[c]

class Color_space():
    def __init__(self):
        self.color_names = None
        self.color_values = None
    def get_color_name(self, i):
        return self.color_names[i]
    def get_color_value(self, i):
        return self.color_values[i]

class LAB(Color_space):
    def __init__(self, accurate=False):
        if accurate:
            self.color_names = np.array(lab_colors)
        else:
            self.color_names = np.unique(lab_colors)
        color_center = []
        for color in np.unique(lab_colors):
            mask = lab_colors == color
            avg = lab_matrix[mask].mean(axis=0)
            color_center.append(avg)
        cie_lab = np.array([color_center])
        cv_lab = np.zeros_like(cie_lab)
        cv_lab[...,0] = cie_lab[...,0] * 255 / 100
        cv_lab[...,1] = cie_lab[...,1] + 128
        cv_lab[...,2] = cie_lab[...,2] + 128
        cv_lab = cv_lab.astype(np.uint8)
        cv_bgr = cv2.cvtColor(cv_lab, cv2.COLOR_LAB2BGR)
        np.squeeze(cv_bgr)
        self.color_values = [(int(b),int(g),int(r)) for b, g, r in np.squeeze(cv_bgr)]
        if accurate:
            self.lab_centers = lab_matrix
        else:
            self.lab_centers = np.squeeze(cie_lab)
Color = LAB(accurate=True)

def get_pose(model, img):    
    pil_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
    pil_img = Image.fromarray(pil_img)    
        
    batch_var = trans(pil_img).unsqueeze(0).cuda()
    with torch.no_grad():
        predicted_outputs, _ = model(batch_var)

    output1, output2 = predicted_outputs[-2], predicted_outputs[-1]
    heatmap = output2.cpu().data.numpy().transpose(0, 2, 3, 1)[0]
    paf = output1.cpu().data.numpy().transpose(0, 2, 3, 1)[0]

    real_shape = img.shape

    heatmap = cv2.resize(heatmap, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    heatmap = heatmap[0:real_shape[0], 0:real_shape[1], :]
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    paf = cv2.resize(paf, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    paf = paf[0:real_shape[0], 0:real_shape[1], :]
    paf = cv2.resize(paf, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
    candidate, subset = decode_pose(
        img, param, heatmap, paf)  
    
    return subset, candidate

def get_sample_points(img, persons, candidate, parts=UPPER):
    samples = []
    pts = []

    for person in persons:
        pts.append((person[IMPORTANT] > 0).sum())

    if len(pts) > 0:        
        idx = np.array(pts).argmax()
        person = persons[idx]  
    
        for part in parts:
            if (person[part] > 0).sum() == len(part):              

                x_coord = np.linspace(candidate[int(person[part[0]])][0], 
                                    candidate[int(person[part[1]])][0], num=10).astype(int)
                y_coord = np.linspace(candidate[int(person[part[0]])][1], 
                                    candidate[int(person[part[1]])][1], num=10).astype(int)                        
                for x, y in zip(x_coord, y_coord):
                    samples.append(img[y, x, :])            
                
    samples = np.array(samples)
    
    return samples

def get_color(orig, samples, color_space):
    h, w = orig.shape[:2]       
    color_center = []
    for c_s in samples:
        delta = delta_e_cie2000(c_s, color_space.lab_centers)
        color_center.append(np.argmin(delta))
    color = stats.mode(color_center)[0][0] 
    return color

def superpixelize(color_img):
    segments = slic(color_img, n_segments=5, compactness=10, sigma=1)
    groups = np.unique(segments)
    vis = np.zeros_like(color_img)
    for g in groups:
        avg = color_img[segments == g, :].mean(axis=0)
        vis[segments == g, :] = avg
    return vis

def get_color_img(img, CODE):
    if CODE == 'HSV':
        trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        trans_img[...,0] *= 2 
        trans_img[...,1] = trans_img[...,1] / 2.55 
        trans_img[...,2] = trans_img[...,2] / 2.55 
    elif CODE == 'LAB':
        trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(float)
        trans_img[...,0] = trans_img[...,0] * 100.0 / 255
        trans_img[...,1] = trans_img[...,1] - 128
        trans_img[...,2] = trans_img[...,2] - 128
    else:
        return None
    return trans_img
    
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst