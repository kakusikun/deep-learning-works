import os
import os.path as osp
import sys
import cv2
import xml.etree.ElementTree as ET
import time
import re
from app.load_openvino import DNet, in_blob, out_blob
from app.load_pose import *

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pandas as pd
import os.path as osp
from tqdm import tqdm

from config.config_manager import _C as model_config
from config.config_manager import _A as app_config
from model.managers.manager_reid_trick import TrickManager
from data.build_transform import build_transform
from tools.utils import deploy_macro, AverageMeter
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
            deploy_macro(model_config)
            self.setIO()
            self.set_DetNet()
            self.set_ReIDNet()
            self.set_PoseNet()
            self.reid_prep = self.pytorch_reid_input_prep()
            self.det_prep = self.openvino_detection_input_prep()

            if not osp.exists(self.a_cfg.SAVE_OUTPUT):
                os.mkdir(self.a_cfg.SAVE_OUTPUT)

    def openvino_detection_input_prep(self):
        def prep(frame):
            inputs = cv2.resize(frame.copy(), (self.a_cfg.DNET.INPUT_SIZE[0], self.a_cfg.DNET.INPUT_SIZE[1]))
            inputs = inputs.transpose(2,0,1)[np.newaxis,:]
            return inputs
        return prep
    
    def pytorch_reid_input_prep(self):
        trans = build_transform(self.m_cfg, is_train=False)
        def prep(data):
            tensors = []
            for c_im in data:
                c_im = cv2.cvtColor(c_im, cv2.COLOR_BGR2RGB) 
                c_im = Image.fromarray(c_im)
                tensors.append(trans(crop_image).unsqueeze(0)) 
            tensors = torch.cat(tensors, dim=0)
            return tensors
        return prep

    def setIO(self, cfg=None):
        self.terminate = False
        if cfg is not None:
            self.a_cfg = cfg

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

    def set_DetNet(self, cfg=None):
        if cfg is not None:
            self.a_cfg = cfg
        if self.a_cfg.DNET.TYPE == 'net':
            self.in_blob = in_blob
            self.out_blob = out_blob
            self.DNet = DNet
        else:
            self.DNet = None            
    
    def set_ReIDNet(self, cfg=None):
        if cfg is not None:
            self.m_cfg = cfg
        manager = TrickManager(self.m_cfg)
        manager.use_multigpu()
        self.ReIDNet = manager.model
        self.ReIDNet.eval()
        

    def set_PoseNet(self):
        self.PoseNet = load_pose_model(self.a_cfg.PNET)

    def get_input(self):
        frame = None
        path = None
        if self.a_cfg.INPUT.TYPE == 'cam' or self.a_cfg.INPUT.TYPE == 'video':
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                self.frame_idx += 1
                if not ret:
                    self.terminate == True        
        else:
            if self.a_cfg.INPUT.TYPE == 'image':
                try:
                    path = next(self.cap)
                    frame = cv2.imread(path)
                    self.frame_idx += 1
                except StopIteration:
                    self.terminate = True
        
        return path, frame
        
    def get_det(self, frame):       
        inputs = self.det_prep(frame)

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
            count = 0
            bboxes = []
            for obj in output[0][0]:
                if obj[2] > 0.5:                    
                    try:
                        x1 = int(obj[3] * initial_w)
                        y1 = int(obj[4] * initial_h)
                        x2 = int(obj[5] * initial_w)
                        y2 = int(obj[6] * initial_h)                        
                        crop_image = frame[y1:y2, x1:x2, :]                        
                        crop_images.append(crop_image)  
                        bboxes.append("{} {} {} {}".format(x1, y1, x2, y2))                       
                        count += 1                        
                    except:
                        continue
        
        return crop_images, count, bboxes

    def get_embedding(self, det):
        if len(det) > 0:
            det = self.reid_prep(det)
            with torch.no_grad():
                feats = self.ReIDNet(det.cuda())
            feats = F.normalize(feats)
            self.group.add_feat(feats)
    
    def get_colors(self, df):
        ucs = []
        lcs = []
        ucis = []
        lcis = []
        for i in df.index:
            uci = -1
            lci = -1
            uc = 'none'
            lc = 'none'

            # read and crop image
            x1, y1, x2, y2 = [i for i in map(int, df.loc[i, 'bbox'].split(" "))]
            img = cv2.imread(df.loc[i, 'img'])
            img = img[y1:y2, x1:x2, :]
            color_img = get_color_img(img, 'LAB')

            # get pose
            start = time.time()
            subset, candidate = get_pose(self.PoseNet, img)     
            end = time.time()
            self.pose_avg_time.update(end-start)

            # get color
            start = time.time()
            sup_img = superpixelize(color_img)
            usamples, upoints = get_sample_points(sup_img, subset, candidate, parts=UPPER)
            lsamples, lpoints = get_sample_points(sup_img, subset, candidate, parts=THIGH)    

            if usamples.shape[0] > 0:
                uci = get_color(usamples, LAB(accurate=True))
                uc = Color.get_color_name(uci)    
                self.visualize(sup_img, upoints, (0,0,255))               
            if lsamples.shape[0] > 0:                
                lci = get_color(lsamples, LAB(accurate=True))     
                lc = Color.get_color_name(lci)     
                self.visualize(sup_img, lpoints, (0,255,0))      
            end = time.time()
            self.color_avg_time.update(end-start)

            ucs.append(uc)
            lcs.append(lc)
            ucis.append(uci)
            lcis.append(lci)

        return ucs, lcs, ucis, lcis
   
    def show_image(self):
        image = self.orig_frame.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image = Image.fromarray(image)
        return image

    def update(self):
        self.people.update(self.group.df)
        self.group = PersonDB()

    def visualize(self, to_plot, points, color):
        for i in range(0, len(points), 2):
            x1 ,y1 = int(points[i]), int(points[i+1])
            cv2.circle(to_plot, (x1, y1), 5, (0,0,0), -1)
            cv2.circle(to_plot, (x1, y1), 4, color, -1)
        
    def render(self, frame):
        to_plot = frame.copy()
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
            path, frame = self.get_input()
            if self.terminate:
                break
            dets, count, bboxes = self.get_det(frame)

            self.group.df.img = [path] * count
            self.group.df.id = lists(range(count))
            self.group.df.bbox = bboxes

            ucs, lcs, ucis, lcis = self.get_colors(self.group.df)  

            self.group.df.uc = ucs
            self.group.df.lc = lcs
            self.group.df.uci = ucis
            self.group.df.lci = lcis
                  
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
        self.df.to_csv("{}".format(path))




if __name__ == '__main__':
    model_config.merge_from_file("/media/allen/mass/deep-learning-works/reid_app.yml")
    app_config.merge_from_file("/media/allen/mass/deep-learning-works/app.yml")
    app = App(model_config, app_config)

    # for branch in os.listdir('/media/allen/mass/recording4/'):   
    #     app_config.SAVE_OUTPUT = '/media/allen/mass/office_color4/{}.csv'.format(branch)
    #     if osp.exists(app_config.SAVE_OUTPUT):
    #         logger.info("{} exists".format(app_config.SAVE_OUTPUT))
    #         continue
    #     app_config.INPUT.PATH = osp.join('/media/allen/mass/recording4/', branch)
    #     app.setIO(app_config)
    #     print(app_config)        
    #     app.build()
    

    df = pd.read_csv("/media/allen/mass/office_color/office_color.csv")

    df = df.drop('Unnamed: 0', axis=1)

    src = "/media/allen/mass/office/office/"
    dst = "/media/allen/mass/office/vis_polygon/"
    if not osp.exists(dst):
        os.mkdir(dst)

    # df.img = df.img.apply(lambda x: osp.join(src, x))    
    
    # Color = LAB(accurate=True)

    pattern = re.compile(r'\[(\d+), (\d+), (\d+), (\d+)\]')
    LAB_Color = LAB(accurate=True)
    ucs = []
    lcs = []
    ucis = []
    lcis = []
    upts = []
    lpts = []
    uptsi = []
    lptsi = []
    indice = []
    for i in tqdm(df.index):
        uci = -1
        lci = -1
        uc = ''
        lc = ''
        indice.append(i)

        # read and crop image
        x1, y1, x2, y2 = [i for i in map(int, pattern.search(df.loc[i, 'bbox']).groups())]    
        img = cv2.imread(osp.join(src, df.loc[i, 'img']))
        img = img[y1:y2, x1:x2, :]
        color_img = get_color_img(img.copy(), 'HLS')

        # get pose
        subset, candidate = get_pose(app.PoseNet, img)    

        # get color
        # sup_img = superpixelize(color_img)
        usamples, upoints = get_sample_points(color_img, subset, candidate, parts=UPPER)
        lsamples, lpoints = get_sample_points(color_img, subset, candidate, parts=THIGH)    

        fname, ext = osp.splitext(df.loc[i, 'img'])        
        fname = "{}_{}{}".format(fname, i, ext)
        print(fname)
        # uplot = lplot = False
        
        if usamples.shape[0] > 0:
            # uci = get_color(usamples, LAB_Color)
            # uc = LAB_Color.get_color_name(uci)   
            h, s, l = get_hsl_median(usamples)
            uc = '{} {} {}'.format(h,s,l) 
            upts.extend(upoints)
            uptsi.extend([i]*len(upoints))
            # app.visualize(img, upoints, (0,0,255))   
            # uplot = True
        if lsamples.shape[0] > 0:                
            # lci = get_color(lsamples, LAB_Color)     
            # lc = LAB_Color.get_color_name(lci) 
            h, s, l = get_hsl_median(lsamples)
            lc = '{} {} {}'.format(h,s,l)  
            lpts.extend(lpoints)
            lptsi.extend([i]*len(lpoints))  
            # app.visualize(img, lpoints, (0,255,0)) 
            # lplot = True
        # except:
        #     print("Stop at index {}".format(i))
        #     upts = np.array(upts)
        #     uptsi = np.array(uptsi)
        #     lpts = np.array(lpts)
        #     lptsi = np.array(lptsi)

        #     np.save(osp.join(dst, 'upts.npy'), upts)
        #     np.save(osp.join(dst, 'uptsi.npy'), uptsi)
        #     np.save(osp.join(dst, 'lpts.npy'), lpts)
        #     np.save(osp.join(dst, 'lptsi.npy'), lptsi)
            
        #     df.uc = ucs
        #     df.lc = lcs
        #     df.uci = ucis
        #     df.lci = lcis
        #     df.to_csv("/media/allen/mass/office_color/office_color_polygon.csv")
        #     sys.exit(1)

        print(i, "---", uc, "---", lc)
        ucs.append(uc)
        lcs.append(lc)
        ucis.append(uci)
        lcis.append(lci)

    upts = np.array(upts)
    uptsi = np.array(uptsi)
    lpts = np.array(lpts)
    lptsi = np.array(lptsi)

    np.save(osp.join(dst, 'upts.npy'), upts)
    np.save(osp.join(dst, 'uptsi.npy'), uptsi)
    np.save(osp.join(dst, 'lpts.npy'), lpts)
    np.save(osp.join(dst, 'lptsi.npy'), lptsi)
    
    df.uc = pd.Series(ucs, index=indice)
    df.lc = pd.Series(lcs, index=indice)
    df.uci = pd.Series(ucis, index=indice)
    df.lci = pd.Series(lcis, index=indice)
    df.to_csv("/media/allen/mass/office_color/office_color_polygon.csv")