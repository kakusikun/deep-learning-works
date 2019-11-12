import os
import os.path as osp
import sys
import cv2
import xml.etree.ElementTree as ET

from app.load_openvino import DNet, in_blob, out_blob

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from sklearn.utils.linear_assignment_ import linear_assignment as linassign

from config.config_manager import _C as model_config
from config.config_manager import _A as app_config
from model.managers.manager_reid_trick import TrickManager
from data.build_transform import build_transform
from tools.utils import deploy_gpu
from tools.logger import setup_logger
import time
logger = setup_logger("/home/allen/", log_name="REID")

VIDEO = ['avi', 'mp4']
PID = 0
DET_TIME = 1
STATUS = 2 # -1:detection, 0:unmatched, 1:matched
BBOX = list(range(3,7))
FEAT = list(range(7,7+512))

class App():
    def __init__(self, model_config, app_config):
        self.m_cfg = model_config
        self.a_cfg = app_config
        self.tracks = PersonDB()
        self.dets = PersonDB()
        self.num_person = 0
        self.frame_idx = 0
        self.terminate = False
        deploy_gpu(model_config)
        self.setIO()
        self.set_DetNet()
        self.set_ReIDNet()
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

    def get_det_input(self):
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
        self.orig_frame = frame
        det_input = cv2.resize(frame.copy(), (self.a_cfg.DNET.INPUT_SIZE[0], self.a_cfg.DNET.INPUT_SIZE[1]))
        return det_input.transpose(2,0,1)[np.newaxis,:]
        
    def get_reid_input(self, det_input):
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
            output = self.DNet.infer(inputs={self.in_blob: det_input})
            output = output[self.out_blob]
            initial_h, initial_w = self.orig_frame.shape[:2]
            temp = []
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
                        temp.append([x1, y1, x2, y2]) 
                    except:
                        continue
            self.dets.add_person(temp)
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    def get_embedding(self, reid_input):
        with torch.no_grad():
            feats = self.ReIDNet(reid_input.cuda())
        feats = F.normalize(feats)
        self.dets.fill_emb(feats)
    

    def get_dist(self):
        if self.tracks.dbsize == 0 or self.dets.dbsize == 0:
            return None
       
        tracks_feat = self.tracks.data[:, FEAT]  
    
        dets_feat = self.dets.data[:, FEAT]

        dist = 1 - F.linear(tracks_feat, dets_feat)

        return dist.cpu().numpy()

    def get_match(self, distmat):
        if distmat is None:
            return None

        result = linassign(distmat)
        return result
        
        
    def update(self, match_pairs):
        if self.dets.dbsize == 0:
            return
        elif self.tracks.dbsize == 0:
            self.tracks.data = self.dets.data
            self.tracks.data[:, PID] = torch.arange(self.dets.dbsize) + 1
            self.tracks.data[:, STATUS] = 2
            self.tracks.data[:, DET_TIME] = time.time()            
            self.num_person = self.tracks.data[:, PID].max().item()
            self.dets = PersonDB()
        else:                        
            matched_t = []
            matched_d = []
            for i, (t, d) in enumerate(match_pairs):
                matched_t.append(t)
                matched_d.append(d)

            unmatched_d = list(set(list(range(self.dets.dbsize))).difference(set(matched_d)))
            unmatched_t = list(set(list(range(self.tracks.dbsize))).difference(set(matched_t)))

            self.tracks.set_value(matched_t, DET_TIME, time.time())
            self.tracks.set_value(matched_t, BBOX, self.dets.data[matched_d][:, BBOX])
            self.tracks.set_value(matched_t, FEAT, (self.tracks.data[matched_t][:, FEAT] + self.dets.data[matched_d][:, FEAT]) / 2)
            self.tracks.set_value(matched_t, STATUS, 2)
            self.tracks.set_value(unmatched_t, STATUS, 1)

            self.dets.set_value(unmatched_d, PID, torch.arange(len(unmatched_d)).cuda() + self.num_person + 1)
            self.dets.set_value(unmatched_d, STATUS, 1)

            self.tracks.add_det(self.dets.data[unmatched_d])
            self.num_person = self.tracks.data[:, PID].max().item()

            self.tracks.remove_old(10)
            self.dets = PersonDB()

    def render(self):
        mask = self.tracks.data[:, STATUS] == 2
        targets = self.tracks.data[mask]
        to_plot = self.orig_frame.copy()
        for i in range(targets.shape[0]):
            tid = targets[i, PID].int().item()
            bbox = targets[i, BBOX].cpu().numpy()
            np.random.seed(tid)
            color = (int(np.random.randint(0,256,1)[0]), int(np.random.randint(0,256,1)[0]), int(np.random.randint(0,256,1)[0]))
            cv2.rectangle(to_plot, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
            cv2.rectangle(to_plot, (bbox[0], bbox[1]), (int(bbox[0]+50), int(bbox[1]+50)), color, -1)
            cv2.putText(to_plot, str(tid), (int(bbox[0]+15), int(bbox[1]+35)), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 2)
            if self.a_cfg.SAVE:
                fname = "{:06}.jpg".format(self.frame_idx)
                fname = osp.join(self.a_cfg.SAVE_OUTPUT, fname)
                cv2.imwrite(fname, to_plot)
        return to_plot

    def start(self):
        logger.info("Starting ...")
        while not self.terminate:
            det_input = self.get_det_input()

            reid_input = self.get_reid_input(det_input)

            self.get_embedding(reid_input)

            distmat = self.get_dist()

            match_pairs = self.get_match(distmat)

            self.update(match_pairs)

            to_plot = self.render()

            cv2.imshow('', to_plot)

            key = cv2.waitKey(1)
            if key == 27:
                self.terminate = True


class PersonDB():
    def __init__(self):
        self.data = torch.Tensor([]).cuda()
    
    @property
    def dbsize(self):
        return self.data.shape[0]
    
    def add_person(self, bbox):
        # pid, 1
        # det_time, 1
        # status, 1
        # x1, y1, x2, y2, 4
        # embedding, 512
        data = torch.zeros((len(bbox),1+1+1+4+512)).cuda()
        data[:, PID] = -1
        data[:, STATUS] = -1
        data[:, DET_TIME] = time.time()
        data[:, BBOX] = torch.Tensor(bbox).cuda()
        self.data = torch.cat([self.data, data])
    
    def add_det(self, dets):
        self.data = torch.cat([self.data, dets])
    
    def fill_emb(self, feats):
        mask = self.data[:, STATUS] == -1
        temp = self.data[mask]
        temp[:, FEAT] = feats
        self.data[mask] = temp
        self.data[:, STATUS] = 0
    
    def remove_old(self, thresh):
        now = time.time()
        mask = self.data[:, DET_TIME] > now - thresh
        self.data = self.data[~mask]
    
    def set_value(self, indice, attr, value):
        temp = self.data[indice]
        temp[:, attr] = value
        self.data[indice] = temp



if __name__ == '__main__':
    model_config.merge_from_file("/media/allen/mass/deep-learning-works/reid_app.yml")
    app_config.merge_from_file("/media/allen/mass/deep-learning-works/app.yml")

    print(app_config)

    app = App(model_config, app_config)
    app.start()