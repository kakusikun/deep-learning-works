from database.datasets import *
import numpy as np
from PIL import Image
from tools.create_centernet_target import keypoints_target

class build_cocokp_dataset(data.Dataset):
    def __init__(self, data, trans=None):
        self.coco = data['handle']
        self.num_classes = data['num_classes']
        self.num_joints = data['num_keypoints']
        self.stride = data['stride']
        self.max_objs = 32
        self.indice = data['indice']
        self.cat_ids = {v: i for i, v in enumerate(self.coco.getCatIds())}
        self.trans = trans

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
        return bbox
   
    def __len__(self):
        return len(self.indice)
        
    def __getitem__(self, index):
        img_id, img_path = self.indice[index]        
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)   
        
        img = Image.open(img_path)
        bboxes = []
        ptss = []
        valid_ptss = []

        for k in range(num_objs):
            ann = anns[k]
            bboxes.append(self._coco_box_to_bbox(ann['bbox']))            
            cls_id = int(self.cat_ids[ann['category_id']])
            pts = np.array(ann['keypoints'], np.float32).reshape(self.num_joints, 3)
            ptss.append([cls_id, pts[:,:2]])
            valid_ptss.append(ptss[:,2])

        # rescale => hflip => tensorize => normalize
        if self.trans is not None:
            img, ss = self.trans(img, bboxes, ptss)

        outsize = (img.size[0] // self.stride, img.size[1] // self.stride)
      
        ret = keypoints_target(bboxes, ptss, valid_ptss, self.max_objs, self.num_classes, self.num_joints, outsize)
        ret['inp'] = img
        ret['c'] = ss['rescale']['c']
        ret['s'] = ss['rescale']['s']
        ret['img_id'] = img_id
        # TODO: move to be a function
        # => CenterNet_build_kptarget(max_objs -> int, 
        #                             num_classes -> int,  
        #                             num_joints -> int,
        #                             outsize -> tuple,
        #                             )
        #########
        

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