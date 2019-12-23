import argparse
import os
pid = os.getpid()
import sys
sys.path.insert(0, 'C:\Users\Administrator\Downloads\caffe-master\caffe-master\Build\Win32\Release\pycaffe')
import caffe
from testing_data.retinaface_decode import RetinafacePostprocess

import dlib
import cv2
import numpy as np
from scipy.spatial.distance import cdist
import time
import psutil
import multiprocessing as mp

from skimage import transform as trans
import numpy as np

ALIGNED_FACE_COORDS = np.array([[38.2946  ,51.6963],
                                [73.5318  ,51.5014],
                                [56.0252  ,71.7366],
                                [41.5493  ,92.3655],
                                [70.7299  ,92.2041]],dtype=np.float32)

GENDER = ['Female', 'Male']
EMOTION = ['-', '=', '+']

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.values = []
        self.avg = 0.0

    def reset(self):
        self.values = []

    def update(self, val, n=1):
        if len(self.values) >= 100:
            self.values.pop(0)
        self.values.append(val)
        self.avg = np.mean(self.values)

def show(ns, lock, args):
    if args.img == "":
        cap = cv2.VideoCapture(args.cam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    img_path = args.img
    draw = args.draw

    print("Camera Starts")
    count = 0
    while True:
        # Read the frame
        if not img_path:
            # print("Camera is streaming ...")
            _, frame = cap.read()    
            if args.build_fid:
                k = cv2.waitKey(30) & 0xff
                if ns.ready and k==32:
                    print("Build Face ID ...")
                    ns.img = frame.copy()
                if len(ns.results) == 1:
                    with lock:
                        print("Save Face ID ...")
                        fid = ns.results[0]['fid']
                        np.save(os.path.join(args.fid_db, "temp.npy"), fid)
                        ns.img = None
                        break  
        else:
            frame = cv2.imread(img_path)
            frame = cv2.resize(frame, (args.width, args.height))

        if not args.build_fid:
            h, w = frame.shape[:2]
            if ns.ready:
                ns.img = frame.copy()   
            if draw:
                img = frame.copy()
                # Display
                if len(ns.results) > 0:
                    with lock:                        
                        for face in ns.results:
                            x, y, w, h = ns.results[face]['bbox']
                            
                            if args.gae:
                                a, g, e = ns.results[face]['gae']
                            else:
                                a = g = e = ""                            
                            for l_x, l_y in ns.results[face]['landmark']:
                                cv2.circle(img, (int(l_x), int(l_y)), 3, (255,255,255), 3)
                            img[-112:,:112,:] = ns.results[face]['aligned']
                            cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
                            if args.fid:
                                name = ns.results[face]['name']
                            else:
                                name = ""
                            cv2.putText(img, "{} {} {} {}".format(a, g, e, name), (int(x), int(y+15)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)

                # if w*h < min_area:
                #     cv2.imwrite('C:\\Users\\Administrator\\Downloads\\facedetection_v2\\dlib_min_{}_{}.jpg'.format(w,h), img)
                #     min_area = w*h
                base = 30
                step = 20
                cv2.putText(img, 'Size   : {}x{}'.format(int(w),int(h))              , (10,base), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(img, 'Size   : {}x{}'.format(int(w),int(h))              , (10,base), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 1)

                cv2.putText(img, 'Det    : {:.2f}ms'.format(ns.time_info['det']*1000)   , (10,base+step), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(img, 'Det    : {:.2f}ms'.format(ns.time_info['det']*1000)   , (10,base+step), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 1)

                cv2.putText(img, 'LandM  : {:.2f}ms'.format(ns.time_info['lm']*1000)   , (10,base+step*2), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(img, 'LandM  : {:.2f}ms'.format(ns.time_info['lm']*1000)   , (10,base+step*2), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 1)

                cv2.putText(img, 'Align  : {:.2f}ms'.format(ns.time_info['al']*1000)   , (10,base+step*3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(img, 'Align  : {:.2f}ms'.format(ns.time_info['al']*1000)   , (10,base+step*3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 1)

                cv2.putText(img, 'GAE    : {:.2f}ms'.format(ns.time_info['gae']*1000)   , (10,base+step*4), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(img, 'GAE    : {:.2f}ms'.format(ns.time_info['gae']*1000)   , (10,base+step*4), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 1)

                cv2.putText(img, 'FID    : {:.2f}ms'.format(ns.time_info['fid']*1000)   , (10,base+step*5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(img, 'FID    : {:.2f}ms'.format(ns.time_info['fid']*1000)   , (10,base+step*5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 1)
                total = ns.time_info['det'] + ns.time_info['lm'] + ns.time_info['al'] + ns.time_info['gae'] + ns.time_info['fid']
                cv2.putText(img, 'Total  : {:.2f}ms'.format(total*1000)   , (10,base+step*6), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(img, 'Total  : {:.2f}ms'.format(total*1000)   , (10,base+step*6), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 1)

                
                cv2.putText(img, 'CPU   : {:.2f}%'.format(ns.time_info['cpu'])              , (10,base+step*7), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(img, 'CPU   : {:.2f}%'.format(ns.time_info['cpu'])              , (10,base+step*7), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 1)
                
                cv2.putText(img, 'Mem   : {:.2f}Mb'.format(ns.time_info['mem'])              , (10,base+step*8), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(img, 'Mem   : {:.2f}Mb'.format(ns.time_info['mem'])              , (10,base+step*8), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 1)


                cv2.imshow('img', img)
            # print(len(self.faces))
            # print(self.det_time.avg * 1000)

            # Stop if escape key is pressed

            else:
                count += 1
                if count % 100 == 0:
                    total = ns.time_info['det'] + ns.time_info['lm'] + ns.time_info['al'] + ns.time_info['gae'] + ns.time_info['fid']
                    print('Size   : {}x{}'.format(w,h))
                    print('Det    : {:.2f}ms'.format(ns.time_info['det']*1000))
                    print('LandM  : {:.2f}ms'.format(ns.time_info['lm']*1000))
                    print('Align  : {:.2f}ms'.format(ns.time_info['al']*1000))
                    print('GAE    : {:.2f}ms'.format(ns.time_info['gae']*1000))
                    print('FID    : {:.2f}ms'.format(ns.time_info['fid']*1000))
                    print('Total  : {:.2f}ms'.format(total*1000))
                    print('CPU    : {:.2f}%'.format(ns.time_info['cpu']))
                    print('Mem    : {:.2f}Mb'.format(ns.time_info['avg']))
                    if len(ns.results) > 0:
                        print(ns.results[0]['bbox'])
                        if args.gae:
                            print(ns.results[0]['gae'])
                        if args.fid:
                            print(ns.results[0]['fid'][0,:10])
                    print("-------------------")
                    print("")
        else:
            cv2.imshow('img', frame)
        
        k = cv2.waitKey(30) & 0xff
        if k==27:
            ns.img = None        
            break  
    print("Camera Stops")

  
def detect(ns, lock, args):
    def load_fid(path, fid_db, fid_db_names):
        fid_paths = os.listdir(path)
        fids = []
        count = 0
        for fid_path in fid_paths:
            if '.npy' in fid_path:
                fids.append(np.load(os.path.join(path, fid_path)))
                fid_db_names[count] = os.path.splitext(fid_path)[0]
                count += 1
        if len(fids) > 0:
            fid_db.extend(np.vstack(fids))

    def get_faces(detector, img, crop_x_offset, crop_w, results):
    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[:, crop_x_offset:(crop_x_offset + crop_w), :]
        dets = detector(img, 0)
        for i in range(len(dets)):
            tl = dets[i].tl_corner()
            br = dets[i].br_corner()
            x1, y1 = tl.x, tl.y
            x2, y2 = br.x, br.y
            results[i] = {}
            results[i]['bbox'] = [x1+crop_x_offset, y1, x2 - x1, y2 - y1]
        
        return img, dets

    def get_landmarks(aligner, img, dets, crop_x_offset, results):
        landmarks = []
        for j, det in enumerate(dets):
            landmark = []
            adjusted_landmark = []
            raw_landmark = aligner(img, det)
            
            for i in [0,2,4]:
                if i == 0 or i == 2:
                    x1 = raw_landmark.part(i).x
                    y1 = raw_landmark.part(i).y
                    x2 = raw_landmark.part(i+1).x
                    y2 = raw_landmark.part(i+1).y
                    landmark.append([(x1+x2)//2,(y1+y2)//2])
                    adjusted_landmark.append([(x1+x2)//2 + crop_x_offset,(y1+y2)//2])
                else:
                    x1 = raw_landmark.part(i).x
                    y1 = raw_landmark.part(i).y
                    landmark.append([x1, y1])
                    adjusted_landmark.append([x1 + crop_x_offset,y1])
            results[j]['landmark'] = adjusted_landmark
            landmarks.append(landmark)
        return landmarks

    def get_aligned_faces(img, landmarks, results):
        aligned_faces = []
        for i, landmark in enumerate(landmarks):
            outputFaceWidth = outputFaceHeight = 112
            leftEyeToLeftEdgeProportion = 0.35
            rightEyeToRightEdgeProportion = 1 - leftEyeToLeftEdgeProportion

            leftEye = np.array(landmark[1])
            rightEye = np.array(landmark[0])
            dY = rightEye[1] - leftEye[1]
            dX = rightEye[0] - leftEye[0]
            angle = np.degrees(np.arctan2(dY, dX))
            currentBetweenEyesLength = np.sqrt((dX ** 2) + (dY ** 2))
            betweenEyesProportion = (rightEyeToRightEdgeProportion - leftEyeToLeftEdgeProportion)
            betweenEyesLength = betweenEyesProportion * outputFaceWidth
            scale = betweenEyesLength / currentBetweenEyesLength

            eyesCenter = (leftEye + rightEye) / 2
            eyesCenter.astype(np.int)
            M = cv2.getRotationMatrix2D(tuple(eyesCenter), angle, scale)
            tX = outputFaceWidth * 0.5
            tY = outputFaceHeight * leftEyeToLeftEdgeProportion
            M[0, 2] += (tX - eyesCenter[0])
            M[1, 2] += (tY - eyesCenter[1])
            (w, h) = (outputFaceWidth, outputFaceHeight)
            alignedImg = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC) 
            aligned_faces.append(alignedImg)
            results[i]['aligned'] = alignedImg
        return aligned_faces

    def FD_preprocess(img, insize):
        h, w = img.shape[:2]
        inh, inw = insize
        if w > h:
            ratio = inh * 1.0 / inw
            outw, outh = w, w * ratio 
            scale = w*1.0/inw           
        else:
            ratio = inw * 1.0 / inh
            outw, outh = h * ratio, h   
            scale = h*1.0/inh  
        canvas = np.zeros((int(outh), int(outw), 3))
        canvas[:h, :w, :] = img
        canvas = canvas.astype(np.uint8)
        canvas = cv2.resize(canvas,(inw, inh))
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return canvas, scale

    def get_faces_and_landmarks(detector, fd_postprocessor, img, insize, crop_x_offset, crop_w, fd_conf, results):
        img = img[:, crop_x_offset:(crop_x_offset + crop_w), :]
        img, scale = FD_preprocess(img, insize)
        caffe_input = np.transpose(img, (2,0,1))[np.newaxis, :]
        detector.blobs['data'].data[...] = caffe_input
        out = detector.forward()
        probs = [out['face_rpn_cls_prob_reshape_stride8'], 
                out['face_rpn_cls_prob_reshape_stride16'], 
                out['face_rpn_cls_prob_reshape_stride32']]
        boxreg = [out['face_rpn_bbox_pred_stride8'], 
                out['face_rpn_bbox_pred_stride16'], 
                out['face_rpn_bbox_pred_stride32']]
        markreg = [out['face_rpn_landmark_pred_stride8'], 
                out['face_rpn_landmark_pred_stride16'], 
                out['face_rpn_landmark_pred_stride32']]
        output = fd_postprocessor.process(probs, boxreg, markreg, confidence=fd_conf, nms=0.4)
        if len(output) > 0:
            for i, bbox in enumerate(output[0][0]):
                bbox *= scale
                x1, y1, x2, y2 = bbox
                results[i] = {}
                results[i]['bbox'] = [x1+crop_x_offset, y1, x2 - x1, y2 - y1]
            for i, _landmark in enumerate(output[0][2]):
                landmark = _landmark.copy()
                landmark *= scale
                landmark[:,0] += crop_x_offset
                results[i]['landmark'] = landmark
            return img, output[0][2]
        else:
            return img, []

    def get_insightface_aligned_faces(get_aligned_matrix_handle, img, landmarks, results):
        aligned_faces = []
        for i, landmark in enumerate(landmarks):
            get_aligned_matrix_handle.estimate(landmark, ALIGNED_FACE_COORDS)
            M = get_aligned_matrix_handle.params[0:2,:]
            alignedImg = cv2.warpAffine(img, M, (112, 112), borderValue = 0.0)
            aligned_faces.append(alignedImg)
            results[i]['aligned'] = alignedImg
        return aligned_faces

    def get_GAE(gae_net, inputs, results):
        for i, _input in enumerate(inputs):
            _input = (_input/255.0 - 0.5)/0.5
            caffe_input = np.transpose(_input, (2,0,1))[np.newaxis, :]
            gae_net.blobs['blob1'].data[...] = caffe_input
            out = gae_net.forward()
            raw_age = out['fc_blob2'].reshape(-1,2)
            raw_gender = out['sigmoid_prob1']
            raw_emotion = out['fc_blob4']
            age = (raw_age[:,0] < raw_age[:,1]).sum()
            gender = GENDER[int(raw_gender[0] >= 0.5)]
            emotion = EMOTION[raw_emotion.argmax()]
            results[i]['gae'] = (age, gender, emotion)

    def get_fid(fid_net, inputs, results):
        for i, _input in enumerate(inputs):
            _input = (_input/255.0 - 0.5)/0.5
            caffe_input = np.transpose(_input, (2,0,1))[np.newaxis, :]
            fid_net.blobs['blob1'].data[...] = caffe_input
            # self.fid_net.blobs['blob2'].data[...] = np.ones((1,16,56,56))
            # self.fid_net.blobs['blob3'].data[...] = np.ones((1,16,28,28))
            # self.fid_net.blobs['blob4'].data[...] = np.ones((1,16,14,14))
            out = fid_net.forward()
            # results[i]['fid'] = out['batch_norm_blob68']
            fid = out['batch_norm_blob50']
            fid /= np.linalg.norm(fid)
            results[i]['fid'] = fid
    
    def get_fids_name(fid_db, fid_db_names, args, results):
        if len(results) > 0:
            if fid_db is not None:
                for face in results:
                    fid = results[face]['fid']
                    cosine_sim = 1 - cdist(fid, fid_db, metric='cosine')
                    idx = np.argmax(cosine_sim)
                    if cosine_sim[idx] > args.fid_conf:
                        results[face]['name'] = fid_db_names[idx]
                    else:
                        results[face]['name'] = "?"
            else:
                for face in results:
                    results[face]['name'] = ""

    if args.dlib_FD:
        print('Dlib FD Loading ...')
        detector = dlib.get_frontal_face_detector()
        print('Dlib Align Loading ...')
        aligner = dlib.shape_predictor(args.dlib_lm_net)
    else:
        print('Caffe FD Loading ...')
        assert args.fd_net != ""
        detector = caffe.Net(args.fd_net, os.path.splitext(args.fd_net)[0] + ".caffemodel",caffe.TEST)
        insize = detector.blobs['data'].data[...].shape[2:]
        fd_postprocessor = RetinafacePostprocess(insize[1], insize[0])
    if args.gae:
        print('Caffe GAE Loading ...')
        gae_net = caffe.Net(args.gae_net, os.path.splitext(args.gae_net)[0] + ".caffemodel",caffe.TEST)
    if args.fid:
        print('Caffe FID Loading ...')
        fid_net = caffe.Net(args.fid_net, os.path.splitext(args.fid_net)[0] + ".caffemodel",caffe.TEST)

    get_aligned_matrix_handle = trans.SimilarityTransform()
    det_time = AverageMeter()
    lm_time = AverageMeter()
    al_time = AverageMeter()
    gae_time = AverageMeter()
    fid_time = AverageMeter()
    py = psutil.Process(pid)
    cpu_usage = AverageMeter()
    mem_usage = AverageMeter()
    crop_x_offset = args.width // 4
    crop_w = args.width // 2

    if args.fid:            
        fid_db = []
        fid_db_names = {}
        load_fid(args.fid_db, fid_db, fid_db_names)

    # Load the cascade
    print("Detection Starts")
    while True:
        if ns.img is not None:
            ns.ready = False
            img = ns.img.copy()                         
            results = {}
            if args.dlib_FD:
                det_start = time.time() 
                img, dets = get_faces(detector, img, crop_x_offset, crop_w, results)
                _det_time = time.time() - det_start

                lm_start = time.time()
                landmarks = get_landmarks(aligner, img, dets, crop_x_offset, results)
                _lm_time = time.time() - lm_start

                al_start = time.time() 
                aligned_faces = get_aligned_faces(img, landmarks, results)
                _al_time = time.time() - al_start
            else:
                det_start = time.time() 
                img, landmarks = get_faces_and_landmarks(detector, fd_postprocessor, img, insize, crop_x_offset, crop_w, args.fd_conf, results)
                _det_time = time.time() - det_start

                _lm_time = 0.0

                al_start = time.time() 
                aligned_faces = get_insightface_aligned_faces(get_aligned_matrix_handle, img, landmarks, results)
                _al_time = time.time() - al_start

            if args.gae:
                gae_start = time.time()
                get_GAE(gae_net, aligned_faces, results)
                _gae_time = time.time() - gae_start
            else:
                _gae_time = 0.0

            if args.fid:
                fid_start = time.time()
                get_fid(fid_net, aligned_faces, results)
                if not args.build_fid:
                    get_fids_name(fid_db, fid_db_names, args, results)
                _fid_time = time.time() - fid_start
            else:
                _fid_time = 0.0
                
            with lock:
                ns.results = results                    
                det_time.update(_det_time)
                lm_time.update(_lm_time)
                al_time.update(_al_time)
                gae_time.update(_gae_time)
                fid_time.update(_fid_time)
                cpu_usage.update(py.cpu_percent()/psutil.cpu_count())
                mem_usage.update(py.memory_info()[0]/2.**20)
                ns.time_info = {'det':det_time.avg, 'lm':lm_time.avg, 'al':al_time.avg, 'gae':gae_time.avg, 'fid':fid_time.avg, 'cpu':cpu_usage.avg, 'mem':mem_usage.avg}
            ns.ready = True
        else:
            break
    print("Detection Stops")
                        
        
# Release the VideoCapture object

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Deep Learning")
    parser.add_argument("--cam", default="", help="camera index", type=int)
    parser.add_argument("--width", default="", help="width of camera resolution", type=int)
    parser.add_argument("--height", default="", help="height of camera resolution", type=int)
    parser.add_argument("--dlib_FD", action='store_true', help="use dlib to detect face")
    parser.add_argument("--gae", action='store_true', help="use gea")
    parser.add_argument("--fid", action='store_true', help="use fid")
    parser.add_argument("--fd_net", default="", help="path to face detection net", type=str)
    parser.add_argument("--dlib_lm_net", default="", help="path to dlib landmark net", type=str)
    parser.add_argument("--gae_net", default="", help="path to face detection net", type=str)
    parser.add_argument("--fid_net", default="", help="path to face ID net", type=str)
    parser.add_argument("--draw", action='store_true', help="draw output")
    parser.add_argument("--img", default="", help="path to img", type=str)
    parser.add_argument("--fd_conf", default=0.5, help="confidence for face detection", type=float)
    parser.add_argument("--fid_conf", default=0.5, help="confidence for face ID", type=float)
    parser.add_argument("--build_fid", action='store_true', help="take a picture and build fid")
    parser.add_argument("--fid_db", default="", help="path to face ID db", type=str)
    args = parser.parse_args()

    mgr = mp.Manager()
    namespace = mgr.Namespace()
    namespace.results = []
    namespace.img = ""
    namespace.ready = True
    namespace.time_info = {'det':0.0, 'lm':0.0, 'al':0.0, 'gae':0.0, 'fid':0.0, 'cpu':0.0, 'mem':0.0}
    
    lock = mp.Lock()

    s = mp.Process(target=show, args=(namespace, lock, args,))
    d = mp.Process(target=detect, args=(namespace, lock, args,))
        
    d.start()
    s.start()

    s.join()
    d.join()