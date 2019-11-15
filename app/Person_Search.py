import os
import os.path as osp
import sys
import cv2
import xml.etree.ElementTree as ET
import time

from app.load_openvino import DNet, in_blob, out_blob

from app.load_pose import *

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pandas as pd

from config.config_manager import _C as model_config
from config.config_manager import _A as app_config
from model.managers.manager_reid_trick import TrickManager
from data.build_transform import build_transform
from tools.utils import deploy_gpu, AverageMeter
from tools.logger import setup_logger
import time
logger = setup_logger("/home/allen/", log_name="REID")

class App():
    def __init__(self, model_config, app_config):
        self.m_cfg = model_config
        self.a_cfg = app_config
        self.people = PersonDB(cache=self.a_cfg.DATABASE)
        self.det_avg_time = AverageMeter()
        self.pose_avg_time = AverageMeter()
        self.color_avg_time = AverageMeter()
        if not self.a_cfg.DATABASE:
            self.group = PersonDB()
            self.frame_idx = 0
            self.terminate = False
            deploy_gpu(model_config)
            self.setIO()
            self.set_DetNet()
            self.set_ReIDNet()
            self.set_PoseNet()
            if not osp.exists(self.a_cfg.SAVE_OUTPUT):
                os.mkdir(self.a_cfg.SAVE_OUTPUT)

    def setIO(self):
        if self.a_cfg.INPUT.TYPE == 'cam':
            device = int(self.a_cfg.INPUT.split("_")[-1])
            self.cap = cv2.VideoCapture(device)
        else:
            assert self.a_cfg.INPUT.PATH != ""
            if self.a_cfg.INPUT.TYPE == 'video':
                self.cap = cv2.VideoCapture(self.a_cfg.INPUT.PATH)
            elif self.a_cfg.INPUT.TYPE == 'image':
                img_paths = sorted([osp.join(root, f) for root, _, files in os.walk(self.a_cfg.INPUT.PATH)
                                                          for f in files if 'jpg' in f or 'png' in f])
                self.cap = iter(img_paths)

    def set_DetNet(self):
        if self.a_cfg.DNET.TYPE == 'net':
            self.in_blob = in_blob
            self.out_blob = out_blob
            self.DNet = DNet
        else:
            self.DNet = None            
    
    def set_ReIDNet(self):
        manager = TrickManager(self.m_cfg)
        manager.use_multigpu()
        self.ReIDNet = manager.model
        self.ReIDNet.eval()
        self.reid_prep = build_transform(self.m_cfg, is_train=False)

    def set_PoseNet(self):
        self.PoseNet = load_pose_model(self.a_cfg.PNET)

    def get_input(self):
        if self.a_cfg.INPUT.TYPE == 'cam' or self.a_cfg.INPUT.TYPE == 'video':
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                self.frame_idx += 1
                if not ret:
                    self.terminate == True        
        else:
            if self.a_cfg.INPUT.TYPE == 'image':
                try:
                    self.path = next(self.cap)
                    frame = cv2.imread(self.path)
                    self.frame_idx += 1
                except StopIteration:
                    self.terminate = True
        
        return frame
        
    def get_det(self, frame):
        self.orig_frame = frame
        inputs = cv2.resize(frame.copy(), (self.a_cfg.DNET.INPUT_SIZE[0], self.a_cfg.DNET.INPUT_SIZE[1]))
        inputs.transpose(2,0,1)[np.newaxis,:]
        crop_images = []
        if self.a_cfg.DNET.TYPE == 'xml':
            anno = osp.splitext(self.path)[0] + ".xml"
            anno = osp.join(self.a_cfg.DNET.PATH, anno)
            tree = ET.parse(anno)
            root = tree.getroot()
            temp = []
            for x in root.findall('object'):
                bbox =x.find('bndbox')
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)
                temp.append([x1, y1, x2, y2])             
            self.dets.add_person(temp)
            for bbox in temp:
                x1, y1, x2, y2 = bbox
                crop_image = self.orig_frame[y1:y2, x1:x2, :]
                crop_images.append(self.reid_prep(crop_image).unsqueeze(0))            
        else:
            start = time.time()
            output = self.DNet.infer(inputs={self.in_blob: inputs})
            end = time.time()
            self.det_avg_time.update(end-start)
            output = output[self.out_blob]
            initial_h, initial_w = self.orig_frame.shape[:2]
            temp = []
            count = 1
            n = self.group.dbsize
            for obj in output[0][0]:
                if obj[2] > 0.5:                    
                    try:
                        x1 = int(obj[3] * initial_w)
                        y1 = int(obj[4] * initial_h)
                        x2 = int(obj[5] * initial_w)
                        y2 = int(obj[6] * initial_h)                        
                        crop_image = self.orig_frame[y1:y2, x1:x2, :]
                        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB) 
                        crop_image = Image.fromarray(crop_image)
                        crop_images.append(self.reid_prep(crop_image).unsqueeze(0))  
                        self.group.add(self.path, column='img', index=n+count)
                        self.group.add(count, column='id', index=n+count)
                        self.group.add([x1, y1, x2, y2], column='bbox', index=n+count)                        
                        count += 1
                        
                    except:
                        continue
        
        return crop_images

    def get_embedding(self, det):
        if len(det) > 0:
            det = torch.cat(det, dim=0)
            with torch.no_grad():
                feats = self.ReIDNet(det.cuda())
            feats = F.normalize(feats)
            self.group.add_feat(feats)
    
    def get_colors(self):
        for i in self.group.df.index:
            uci = -1
            lci = -1
            uc = 'none'
            lc = 'none'
            x1, y1, x2, y2 = self.group.df.loc[i, 'bbox']
            img = self.orig_frame[y1:y2, x1:x2, :]
            color_img = get_color_img(img, 'LAB')
            start = time.time()
            subset, candidate = get_pose(self.PoseNet, img)     
            end = time.time()
            self.pose_avg_time.update(end-start)

            start = time.time()
            sup_img = superpixelize(color_img)
            usamples, upoints = get_sample_points(sup_img, subset, candidate, parts=UPPER)
            if usamples.shape[0] > 0:
                uci = get_color(img, usamples, upoints)
                uc = Color.get_color_name(uci)                
            
            lsamples, lpoints = get_sample_points(sup_img, subset, candidate, parts=THIGH)
            if lsamples.shape[0] > 0:                
                lci = get_color(img, lsamples, lpoints)     
                lc = Color.get_color_name(lci)           
            end = time.time()
            self.color_avg_time.update(end-start)

            self.group.add(uci, column='uci', index=i)
            self.group.add(lci, column='lci', index=i)
            self.group.add(uc, column='uc', index=i)
            self.group.add(lc, column='lc', index=i)
   
    def show_image(self):
        image = self.orig_frame.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image = Image.fromarray(image)
        return image

    def update(self):
        self.people.update(self.group.df)
        self.group = PersonDB()

        
    def render(self):
        to_plot = self.orig_frame.copy()
        for i in self.group.df.index:
            bbox = self.group.df.loc[i, 'bbox']
            uci = self.group.df.loc[i, 'uci']
            lci = self.group.df.loc[i, 'lci']
            if uci >= 0:
                ucolor = Color.get_color_value(uci)
                cv2.rectangle(to_plot, (bbox[0], bbox[1]), (int(bbox[0]+25), int(bbox[1]+25)), ucolor, -1)
                cv2.rectangle(to_plot, (bbox[0], bbox[1]), (int(bbox[0]+25), int(bbox[1]+25)), (255,255,255), 3)
            if lci >= 0:
                lcolor = Color.get_color_value(lci)
                cv2.rectangle(to_plot, (int(bbox[2]-25), int(bbox[3]-25)), (int(bbox[2]), int(bbox[3])), lcolor, -1)
                cv2.rectangle(to_plot, (int(bbox[2]-25), int(bbox[3]-25)), (int(bbox[2]), int(bbox[3])), (255,255,255), 3)

            cv2.rectangle(to_plot, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,0), 3)           
            
            if self.a_cfg.SAVE:
                fname = "{:06}.jpg".format(self.frame_idx)
                fname = osp.join(self.a_cfg.SAVE_OUTPUT, fname)
                cv2.imwrite(fname, to_plot)
        return to_plot

    def build(self):
        logger.info("Starting ...")
        count = 0
        while True:
            frame = self.get_input()
            if self.terminate:
                break
            det = self.get_det(frame)
            self.get_colors()        
            self.update()
            count += 1
            logger.info("Det:{:.3f}s, Pose:{:.3f}s, Color:{:.3f}".format(self.det_avg_time.avg, self.pose_avg_time.avg, self.color_avg_time.avg))
            if count % 100 == 0:
                logger.info("{:>10} samples is processed".format(count))
                self.people.save(self.a_cfg.SAVE_OUTPUT)
                self.people = PersonDB()
        

class PersonDB():
    def __init__(self, cache=False):
        if not cache:
            self.df = pd.DataFrame(columns=['img', 'id', 'bbox', 'uc', 'uci', 'lc', 'lci'])
        else:
            self.df = pd.read_csv(cache)
        
        self.feat = None
    
    @property
    def dbsize(self):
        return self.df.shape[0]
    
    def add(self, data, index, column):
        if index not in self.df.index:
            df = pd.DataFrame(index=[index], columns=[column])
            df.loc[index, column] = data
            self.df = self.df.append(df, sort=False)
        else:
            self.df.loc[index, column] = data
    
    def add_feat(self, feat):
        if self.feat is None:
            self.feat = feat
        else:
            self.feat = np.vstack(self.feat, feat)

    def update(self, df):
        if df.shape[0] != 0:
            if self.dbsize == 0:
                self.df = df
            else:
                self.df = self.df.append(df, sort=False)
    def save(self, path):
        self.df.to_csv("{}/people.csv".format(path))




if __name__ == '__main__':
    model_config.merge_from_file("/media/allen/mass/deep-learning-works/reid_app.yml")
    app_config.merge_from_file("/media/allen/mass/deep-learning-works/app.yml")

    print(app_config)

    app = App(model_config, app_config)
    app.build()