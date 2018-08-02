#coding=utf-8
import numpy as np
import cv2 
import json
from caffe2.python import workspace

from core.config import merge_cfg_from_file
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import core.test_engine as infer_engine

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

class Model():
    
    def __init__(self,cfg_path,weights_path):
    
        #nms_same_class 0.3  ----  *.yaml/TEST:NMS:0.3 中设置 defalut 0.5
        
        self.gpu_id = 0               #gpu_id default 0
        
        self.score_thresh = 0.4       #score > score_thresh  default 0.3  
        
        self.per_class_thresh = False         #score > class_score_thresh
        self.autotruck_score_thresh = 0.6
        self.forklift_score_thresh = 0.65
        self.digger_score_thresh = 0.65
        self.car_score_thresh = 0.45
        self.bus_score_thresh = 0.0
        self.tanker_score_thresh = 0.55
        self.person_score_thresh = 0.35
        self.minitruck_score_thresh = 0.0
        self.minibus_score_thresh = 0.59
        
        self.class_nms_thresh = 0.85   #nms_between_classes  IOU > class_nms_thresh    default 0.9 
        merge_cfg_from_file(cfg_path)
        self.model = infer_engine.initialize_model_from_cfg(weights_path,self.gpu_id)
        self.dummy_coco_dataset = dummy_datasets.get_steal_oil_class10_dataset()
        print ("model is ok")

    def predict(self,im):

        #class_str_list = []
        data_list = []
        
        with c2_utils.NamedCudaScope(self.gpu_id):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(self.model, im, None, None
            )
            
        #get box classes
        if isinstance(cls_boxes, list):
            boxes, segms, keypoints, classes = self.convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)
        if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < self.score_thresh:
            return data_list
        #get score
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)
        
        
        #no nms between classes
        '''im1 = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        result1= im1.copy()
        for i in sorted_inds:
            
            bbox = boxes[i, :4]
            score = boxes[i, -1]
            if score < self.score_thresh:
                continue
            #get class-str
            class_str = self.get_class_string(classes[i], score, self.dummy_coco_dataset)            
            cv2.rectangle(result1,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(255,255,0),1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            ((txt_w, txt_h), _) = cv2.getTextSize(class_str, font, 0.35, 1)            
            txt_tl = int(bbox[0]), int(bbox[1]) - int(0.3 * txt_h)
            cv2.putText(result1, class_str, txt_tl, font, 0.35, (218, 227, 218), lineType=cv2.LINE_AA)
            txt_tl = int(bbox[0])+txt_w, int(bbox[1]) - int(0.3 * txt_h)
            cv2.putText(result1, ('%.2f' % score), txt_tl, font, 0.35, (218, 227, 218), lineType=cv2.LINE_AA)
        cv2.imwrite("test1.jpg", result1)'''
        
        #nms between classes
        #im2 = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        #result2= im2.copy()        
        if (len(sorted_inds) > 0):        
            nmsIndex = self.nms_between_classes(boxes, self.class_nms_thresh)  #阈值为0.9，阈值越大，过滤的越少 
            for i in xrange(len(nmsIndex)):                
                bbox = boxes[nmsIndex[i], :4]
                score = boxes[nmsIndex[i], -1]
                if score < self.score_thresh:
                    continue
                #get class-str
                class_str = self.get_class_string(classes[nmsIndex[i]], score, self.dummy_coco_dataset)
                
                #score thresd per class
                if self.per_class_thresh:
                    if 'autotruck' == class_str and score < self.autotruck_score_thresh:
                        continue
                    if 'forklift'  == class_str and score < self.forklift_score_thresh:
                        continue
                    if 'digger'    == class_str and score < self.digger_score_thresh:
                        continue
                    if 'car'       == class_str and score < self.car_score_thresh:
                        continue
                    if 'bus'       == class_str and score < self.bus_score_thresh:
                        continue
                    if 'tanker'    == class_str and score < self.tanker_score_thresh:
                        continue
                    if 'person'    == class_str and score < self.person_score_thresh:
                        continue
                    if 'minitruck' == class_str and score < self.minitruck_score_thresh:
                        continue
                    if 'minibus'   == class_str and score < self.minibus_score_thresh:
                        continue
                
                single_data = {"cls":class_str,"score":float('%.2f' % score),"bbox":{"xmin":int(bbox[0]),"ymin":int(bbox[1]),"xmax":int(bbox[2]),"ymax":int(bbox[3])}}
                data_list.append(single_data)        
        
                '''cv2.rectangle(result2,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(255,255,0),1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                ((txt_w, txt_h), _) = cv2.getTextSize(class_str, font, 0.55, 1)            
                txt_tl = int(bbox[0]), int(bbox[1]) - int(0.3 * txt_h)
                cv2.putText(result2, class_str, txt_tl, font, 0.55, (218, 227, 218), lineType=cv2.LINE_AA)
                txt_tl = int(bbox[0])+txt_w, int(bbox[1]) - int(0.3 * txt_h)
                cv2.putText(result2, ('%.2f' % score), txt_tl, font, 0.35, (218, 227, 218), lineType=cv2.LINE_AA)'''
        #cv2.imwrite("test2.jpg", result2)
        
        #construcrion - data_list
        return data_list
        
    def convert_from_cls_format(self,cls_boxes, cls_segms, cls_keyps):
        """Convert from the class boxes/segms/keyps format generated by the testing
        code.
        """
        box_list = [b for b in cls_boxes if len(b) > 0]
        if len(box_list) > 0:
            boxes = np.concatenate(box_list)
        else:
            boxes = None
        if cls_segms is not None:
            segms = [s for slist in cls_segms for s in slist]
        else:
            segms = None
        if cls_keyps is not None:
            keyps = [k for klist in cls_keyps for k in klist]
        else:
            keyps = None
        classes = []
        for j in range(len(cls_boxes)):
            classes += [j] * len(cls_boxes[j])
        return boxes, segms, keyps, classes
        
    def get_class_string(self,class_index, score, dataset):
        class_text = dataset.classes[class_index] if dataset is not None else \
            'id{:d}'.format(class_index)
        #return class_text + ' {:0.2f}'.format(score).lstrip('0')
        return class_text
    def nms_between_classes(self,boxes, threshold):
        if boxes.size==0:
            return np.empty((0,3))
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        s = boxes[:,4]
        area = (x2-x1+1) * (y2-y1+1)
        I = np.argsort(s)
        pick = np.zeros_like(s, dtype=np.int16)
        counter = 0
        while I.size>0:
            i = I[-1]
            pick[counter] = i
            counter += 1
            idx = I[0:-1]
            xx1 = np.maximum(x1[i], x1[idx])
            yy1 = np.maximum(y1[i], y1[idx])
            xx2 = np.minimum(x2[i], x2[idx])
            yy2 = np.minimum(y2[i], y2[idx])
            w = np.maximum(0.0, xx2-xx1+1)
            h = np.maximum(0.0, yy2-yy1+1)
            inter = w * h        
            o = inter / (area[i] + area[idx] - inter)
            I = I[np.where(o<=threshold)]
        pick = pick[0:counter]  #返回nms后的索引
        return pick
    
    