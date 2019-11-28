import pandas as pd
from PIL import Image, ImageDraw
import cv2
import re
import numpy as np
import sys
import os
import os.path as osp
from tools import bcolors

def display_hsl_quantile(samples):
    h_ths = 15
    l_ths = 26
    samples = samples[samples[:,0] > 0]
    for q in range(0,11,2):
        q *= 0.1
        h = np.quantile(samples[:,0], q=q).astype(int)
        hu = (h + h_ths) % 180
        hl = (h - h_ths) % 180
        if hu < hl:
            h_mask = ((samples[:,0] < hu) * (samples[:,0] >= 0) + (samples[:,0] > hl) * (samples[:,0] <= 180)).astype(bool)
        else:
            h_mask = ((samples[:,0] < hu) * (samples[:,0] > hl)).astype(bool)
            
        l = np.quantile(samples[h_mask,2], q=q).astype(int)
        l_u = np.clip(l + l_ths, 0, 255)
        l_l = np.clip(l - l_ths, 0, 255)
        l_mask = ((samples[:,2] > l_l) * (samples[:,2] < l_u) * h_mask).astype(bool)

        s = np.quantile(samples[l_mask,1], q=q).astype(int)
        print("In samples, the {}{:>2}th quantile{} of HSL is {}{:>3} {:>3} {:>3}{}".format(bcolors.OKGREEN, int(q*10), bcolors.RESET, bcolors.OKGREEN,int(h*2), int(s/255*100), int(l/255*100), bcolors.RESET))

def get_candidate_mask(color, h, s, l):
    hu = (h/2 + 15) % 180
    hl = (h/2 - 15) % 180
    lu = np.clip(l/100*255 + 26, 0, 255)
    ll = np.clip(l/100*255 - 26, 0, 255)
    su = np.clip(s/100*255 + 51, 0, 255)
    sl = np.clip(s/100*255 - 51, 0, 255)
    if hu < hl:
        h_mask = ((color[:,0] < hu) * (color[:,0] >= 0) + (color[:,0] > hl) * (color[:,0] <= 180)).astype(bool)
    else:
        h_mask = ((color[:,0] < hu) * (color[:,0] > hl)).astype(bool)
    s_mask = ((color[:,1] < su) * (color[:,1] > sl)).astype(bool)
    l_mask = ((color[:,2] < lu) * (color[:,2] > ll)).astype(bool)

    mask = h_mask * l_mask * s_mask
    return mask

def visualize(to_plot, points, color):
    for i in range(0, len(points), 2):
        x1 ,y1 = int(points[i]), int(points[i+1])
        cv2.circle(to_plot, (x1, y1), 5, (0,0,0), -1)
        cv2.circle(to_plot, (x1, y1), 4, color, -1)

def get_group(mask, upts, uptsi, lpts, lptsi, df, src, dst):
    pattern = re.compile(r'\[(\d+), (\d+), (\d+), (\d+)\]')
    for i in df[mask].index:
        upt = upts[uptsi==i]
        lpt = lpts[lptsi==i]
        path = df.loc[i, 'img']
        pid = df.loc[i, 'id']
        x1, y1, x2, y2 = [i for i in map(int, pattern.search(df.loc[i, 'bbox']).groups())]
        img = cv2.imread(osp.join(src, path))
        crop = img[y1:y2, x1:x2, :]
        visualize(crop, upt, (0,255,0))
        visualize(crop, lpt, (0,0,255))
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
        fname, ext = osp.splitext(path)    
        fname = "{}_{}{}".format(fname, pid, ext)        
        fname = osp.join(dst, fname)
        cv2.imwrite(fname, merge_img)


if __name__ == '__main__':
    csv_src = sys.argv[1]
    img_src = sys.argv[2]
    img_dst = osp.dirname(csv_src)

    df = pd.read_csv(csv_src, index_col=[0], dtype={'img':str, 'bbox':str, 'id':int, 'uc':str, 'lc':str, 'uci':int, 'lci':int})

    df.uc = df.uc.fillna('-1 -1 -1')
    df.lc = df.lc.fillna('-1 -1 -1')

    df.uc = df.uc.apply(lambda x: np.array([i for i in map(int, x.split(" "))]))
    df.lc = df.lc.apply(lambda x: np.array([i for i in map(int, x.split(" "))]))

    ucs = np.vstack(df.uc.values)
    lcs = np.vstack(df.lc.values)

    uptsi = np.load(osp.join(osp.dirname(csv_src), 'uptsi.npy'))
    upts  = np.load(osp.join(osp.dirname(csv_src), 'upts.npy'))
    lptsi = np.load(osp.join(osp.dirname(csv_src), 'lptsi.npy'))
    lpts  = np.load(osp.join(osp.dirname(csv_src), 'lpts.npy'))

    action = ['']
    while action[0] != '-1':
        print("For upper body color")
        display_hsl_quantile(ucs)
        print("For lower body color")
        display_hsl_quantile(lcs)
        action = input("Enter HSL, H(0~360), S(0~100), L(0~100), example: u 100-50-50 : ").lower().strip().split(" ")
        is_valid = False
        if len(action) == 1:
            if action[0] == '-1':
                print("{}Exit{}".format(bcolors.OKGREEN, bcolors.RESET))
            else:
                print("{}Wrong Condition{}".format(bcolors.WARNING, bcolors.RESET))
        elif len(action) == 2:
            part = action[0]
            try:
                request = np.array([i for i in map(int, action[1].split('-'))])
                assert request.any() >= 0
                assert request[0] <= 360
                assert request[1] <= 100
                assert request[2] <= 100
                is_valid = True
            except:
                print("{}Wrong Condition{}".format(bcolors.WARNING, bcolors.RESET))

            if is_valid:
                if part == 'u':
                    _dst = osp.join(img_dst, part, action[1])
                    if not osp.exists(_dst):
                        os.makedirs(_dst)
                    h, s, l = request
                    u_mask = get_candidate_mask(ucs, h, s, l)
                    print("Searching space size : {}{}{}".format(bcolors.BOLD, u_mask.sum(), bcolors.RESET))
                    print("Result is placed at {}".format(_dst))
                    get_group(u_mask, upts, uptsi, lpts, lptsi, df, img_src, _dst)
                elif part == 'l':
                    _dst = osp.join(img_dst, part, action[1])
                    if not osp.exists(_dst):
                        os.makedirs(_dst)
                    h, s, l = request
                    l_mask = get_candidate_mask(lcs, h, s, l)
                    print("Searching space size : {}{}{}".format(bcolors.BOLD, u_mask.sum(), bcolors.RESET))
                    print("Result is placed at {}".format(_dst))
                    get_group(l_mask, upts, uptsi, lpts, lptsi, df, img_src, _dst)
                else:
                    print("{}Wrong Condition{}".format(bcolors.WARNING, bcolors.RESET))
            
            
        elif len(action) == 4:
            upart = action[0]
            lpart = action[2]
            try:
                urequest = np.array([i for i in map(int, action[1].split('-'))])
                lrequest = np.array([i for i in map(int, action[3].split('-'))])
                assert urequest.any() >= 0
                assert urequest[0] <= 360
                assert urequest[1] <= 100
                assert urequest[2] <= 100
                assert lrequest.any() >= 0
                assert lrequest[0] <= 360
                assert lrequest[1] <= 100
                assert lrequest[2] <= 100
                is_valid = True
            except:
                print("{}Wrong Condition{}".format(bcolors.WARNING, bcolors.RESET))
            if is_valid:
                if upart == 'u' and lpart == 'l':
                    _dst = osp.join(img_dst, upart + "_" + lpart, action[1] + "_" + action[3])
                    if not osp.exists(_dst):
                        os.makedirs(_dst)
                    h, s, l = urequest
                    u_mask = get_candidate_mask(ucs, h, s, l)
                    h, s, l = lrequest
                    l_mask = get_candidate_mask(lcs, h, s, l)
                    print("Searching space size : {}{}{}".format(bcolors.BOLD, u_mask.sum(), bcolors.RESET))
                    print("Result is placed at {}".format(_dst))
                    get_group(l_mask*u_mask, upts, uptsi, lpts, lptsi, df, img_src, _dst)           
                else:
                    print("{}Wrong Condition{}".format(bcolors.WARNING, bcolors.RESET))
            
        else:
            print("{}Wrong Condition{}".format(bcolors.WARNING, bcolors.RESET))
