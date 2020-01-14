from database.datasets import *
import numpy as np
import cv2
import math
from tools.image import get_affine_transform, affine_transform, draw_umich_gaussian, gaussian_radius, color_aug

class build_cocokp_dataset(data.Dataset):
    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, data_coco, data, split, output_stride):
        self.coco = data_coco
        self.num_classes = len(self.coco.loadCats(self.coco.getCatIds()))
        self.num_joints = len(self.coco.loadAnns(ids=self.coco.getAnnIds(catIds=[1]))[0]['keypoints']) // 3
        self.max_objs = 32
        self.default_res = (512, 512)    
        self.output_stride = output_stride
        self.images = data
        self.split = split
        self.cat_ids = {v: i for i, v in enumerate(self.coco.getCatIds())}
        self._data_rng = np.random.RandomState(123)

        # TODO: deprecated
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([[-0.58752847, -0.69563484,  0.41340352],
                                  [ -0.5832747,  0.00994535, -0.81221408],
                                  [-0.56089297,  0.71832671,  0.41158938]], dtype=np.float32)
        # TODO: move to transform => kp_flip(num_joints)
        

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    # TODO: move to transform => RandScale(scale => tuple : (min, max))
    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i
    
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        img_id, img_path = self.images[index]        
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)   
        
        img = cv2.imread(img_path)

        # TODO: move to transform => RandScale(scale -> tuple, output_stride -> int)
        #########
        height, width = img.shape[0], img.shape[1]    
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0

        input_h, input_w = self.default_res
        # image position augmentation
        # scale
        flipped = False
        if self.split == 'train':
            s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = self._get_border(64, img.shape[1])
            h_border = self._get_border(64, img.shape[0])
            # location
            c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
            c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)            
        #########

        # TODO: move to transform => kp_flip(num_joints)
        #########
            # hflip
            if np.random.random() < 0.5:
                flipped = True
                img = img[:, ::-1, :]
                c[0] =  width - c[0] - 1            
        #########


        # processing
        # TODO: move to transform => RandScale(scale -> tuple, output_stride -> int)
        #########
        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        #########

        # TODO: move to transform => normalize
        #########
        # rescale image pixel value to [0,1]
        inp = (inp.astype(np.float32) / 255.)
        # image color augmentation       
        # if self.split == 'train' and np.random.random() < 0.5:
        #     color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        # image normalize
        inp = (inp - self.mean) / self.std
        #########


        # HWC => CHW
        inp = inp.transpose(2, 0, 1)

        # TODO: move to transform => RandScale(scale -> tuple, output_stride -> int)
        #########
        output_h = input_h // self.output_stride
        output_w = input_w // self.output_stride

        num_classes = self.num_classes
        num_joints = self.num_joints
        # transform that applying to gt
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])
        #########


        # TODO: move to be a function
        # => CenterNet_build_kptarget(max_objs -> int, 
        #                             num_classes -> int,  
        #                             num_joints -> int,
        #                             outsize -> tuple,
        #                             )
        #########
        # center, object heatmap
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        # center, keypoint heatmap
        hm_hp = np.zeros((num_joints, output_h, output_w), dtype=np.float32)

        # object size
        wh             = np.zeros((self.max_objs, 2), dtype=np.float32)
        # keypoint location relative to center
        kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)
        # object offset
        reg            = np.zeros((self.max_objs             , 2             ), dtype=np.float32)       
        ind            = np.zeros((self.max_objs             ), dtype=np.int64)
        reg_mask       = np.zeros((self.max_objs             ), dtype=np.uint8)                       
        kps_mask       = np.zeros((self.max_objs             , self.num_joints * 2), dtype=np.uint8)
        hp_offset      = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
        hp_ind         = np.zeros((self.max_objs * num_joints), dtype=np.int64)
        hp_mask        = np.zeros((self.max_objs * num_joints), dtype=np.int64)

        draw_gaussian = draw_umich_gaussian

        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            
            cls_id = int(self.cat_ids[ann['category_id']])
            pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3)
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                pts[:, 0] = width - pts[:, 0] - 1
                for e in self.flip_idx[cls_id]:
                    e_offset = self.flip_idx_offset[cls_id]
                    pts[e[0]+e_offset], pts[e[1]+e_offset] = pts[e[1]+e_offset].copy(), pts[e[0]+e_offset].copy()
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)            
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]            
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1  
                num_kpts = pts[:, 2].sum()
                if num_kpts == 0:
                    hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
                    reg_mask[k] = 0

                hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                hp_radius = max(0, int(hp_radius)) 
                for j in range(num_joints):
                    if pts[j, 2] > 0:
                        pts[j, :2] = affine_transform(pts[j, :2], trans_output)
                        if pts[j, 0] >= 0 and pts[j, 0] < output_w and pts[j, 1] >= 0 and pts[j, 1] < output_h:
                            kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                            kps_mask[k, j * 2: j * 2 + 2] = 1
                            pt_int = pts[j, :2].astype(np.int32)
                            hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
                            hp_ind[k * num_joints + j] = pt_int[1] * output_w + pt_int[0]
                            hp_mask[k * num_joints + j] = 1
                            draw_gaussian(hm_hp[j], pt_int, hp_radius)
                draw_gaussian(hm[cls_id], ct_int, radius)
        ret = {'inp': inp,
               'hm': hm, 'wh':wh, 'reg':reg,
               'reg_mask': reg_mask, 'ind': ind,
               'hm_hp': hm_hp, 'hps': kps, 'hps_mask': kps_mask, 'hp_reg': hp_offset,
               'hp_ind': hp_ind, 'hp_mask': hp_mask,
               'img_id': img_id, 'c': c, 's': s}
        return ret

    @classmethod
    def preprocess(self, img, scales, size, is_flip=False):       
        # height, width = img.shape[0], img.shape[1]    
        height, width = img.shape[0], img.shape[1]    
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        _s = max(img.shape[0], img.shape[1]) * 1.0
        # input_h, input_w = self.default_res
        w, h = size
        input_w = int(np.ceil(w/128)*128)
        input_h = int(np.ceil(h/128)*128)

        # image position augmentation
        # scale
        batch = []
        cbatch = []
        sbatch = []
        
        for sf in scales:   
            s = _s * sf
            if is_flip:
                img = img[:, ::-1, :]
                c[0] =  width - c[0] - 1    
            # processing
            trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
            inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
            # rescale image pixel value to [0,1]
            inp = (inp.astype(np.float32) / 255.)
            # # image normalize
            inp = (inp - self.mean) / self.std
            # HWC => CHW
            inp = inp.transpose(2, 0, 1)
            batch.append(inp)
            cbatch.append(c)
            sbatch.append(s)
        return {'inputs': np.array(batch), 'c': np.array(cbatch), 's': np.array(sbatch)}