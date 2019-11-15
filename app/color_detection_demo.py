import pandas as pd
from PIL import Image, ImageDraw
import cv2
import re
import numpy as np
import sys
import os
import os.path as osp

def get_group(_df, dst):
    pattern = re.compile(r'\[(\d+), (\d+), (\d+), (\d+)\]')
    for idx in range(_df.shape[0]):
        path = _df.iloc[idx].img
        pid = _df.iloc[idx].id
        x1, y1, x2, y2 = [i for i in map(int, pattern.search(_df.iloc[idx].bbox).groups())]
        img = cv2.imread(path)
        crop = img[y1:y2, x1:x2, :]
        c_h, c_w = crop.shape[:2]
        i_h, i_w = img.shape[:2]
        c_r = 300.0 / c_h
        c_w = int(c_w * c_r)
        i_r = 300.0 / i_h
        i_w = int(i_w * i_r)
        crop = cv2.resize(crop, (c_w, 300))
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), 3)
        img = cv2.resize(img, (i_w, 300))
        merge_img = np.zeros((300, c_w+i_w, 3))
        merge_img[:, :c_w, :] = crop
        merge_img[:, c_w:, :] = img
        merge_img = merge_img.astype(np.uint8)
        path = osp.basename(path)
        fname, ext = osp.splitext(path)    
        fname = "{}_{}{}".format(fname, pid, ext)        
        fname = osp.join(dst, fname)
        cv2.imwrite(fname, merge_img)

if __name__ == '__main__':
    src = sys.argv[1]
    dst = osp.dirname(src)

    df = pd.read_csv(src, index_col=[0], dtype={'img':str, 'bbox':str, 'id':int, 'uc':str, 'lc':str, 'uci':int, 'lci':int})
    for c in df.uc.unique():
        if c != 'none':
            print("All detected upper color : {}".format(c))
    for c in df.lc.unique():
        if c != 'none':
            print("All detected lower color : {}".format(c))
    action = ['']
    while action[0] != '-1':
        action = input("Enter filtered condition: ").lower().strip().split(" ")
        if len(action) == 1:
            if action[0] == '-1':
                print("Exit")
            else:
                print("Wrong Condition")
        elif len(action) == 2:
            part = action[0]
            request = action[1]
            if part == 'u':
                _dst = osp.join(dst, part, request)
                if not osp.exists(_dst):
                    os.makedirs(_dst)
                _df = df[df.uc==request]
                print("Searching space size : {}".format(len(_df)))
                print("Result is placed at {}".format(_dst))
                get_group(_df, _dst)
            elif part == 'l':
                _dst = osp.join(dst, part, request)
                if not osp.exists(_dst):
                    os.makedirs(_dst)
                _df = df[df.lc==request]
                print("Searching space size : {}".format(len(_df)))
                print("Result is placed at {}".format(_dst))
                get_group(_df, _dst)
            else:
                print("Wrong Condition")
        elif len(action) == 4:
            upart = action[0]
            urequest = action[1]
            lpart = action[2]
            lrequest = action[3]
            if upart == 'u' and lpart == 'l':
                _dst = osp.join(dst, upart + "_" + lpart, urequest + "_" + lrequest)
                if not osp.exists(_dst):
                    os.makedirs(_dst)
                _df = df[(df.uc==urequest) & (df.lc==lrequest)]
                print("Searching space size : {}".format(len(_df)))
                print("Result is placed at {}".format(_dst))
                get_group(_df, _dst)            
            else:
                print("Wrong Condition")
        else:
            print("Wrong Condition")
