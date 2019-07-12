# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
detect and track 
no flag
"""
import sys
import colorsys
import os
from timeit import default_timer as timer
import time
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import cv2
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

return_boxes = []
global tracker
global stracker

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/tiny-yolo.h5',
        "anchors_path": 'model_data/tiny_yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

       
        self.yolo_model = load_model(model_path, compile=False)
        
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        #start = timer()
        global return_boxes
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        
        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        
        
        
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if predicted_class !='person':
                #del return_boxes[:]
                continue
            elif predicted_class == 'person':
                del return_boxes[:]            
            box = out_boxes[i]
            
            x = int(box[1])
            y = int(box[0])
            w = int(box[3]-box[1])
            h = int(box[2]-box[0])
            if x < 0:
                w = w + x
                x = 0
            if y < 0:
                h = h +y 
                y = 0
                
            bboxx = (x,y,w,h)
            return_boxes.append(bboxx)
        return return_boxes 

    

    def single_tracker(self,tracker_type):
        global tracker       
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == 'CSRT':
            tracker = cv2.TrackerCSRT_create()
        else:
            tracker = None    

        return tracker
    def close_session(self):
        self.sess.close()  

def compute_iou(rec1, rec2):
    S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    sum_area = S1 + S2

    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])

    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)


        
def detect_video(yolo, video_path, output_path=""):
    global return_boxes
    global tracker
    global stracker
    import cv2
    global avg_roi
    vid = cv2.VideoCapture(video_path)
    #vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writeVideo_flag = True if output_path != "" else False
    if writeVideo_flag:
        #print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    
    tracker_types = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[7]
    detect_num = 0
    sum_fps = 0
    #print('type-sum_fps', type(sum_fps))
    avg_fps = 0
    #fps = 0.0
    i = 0
    a = 1
    ST = 0.5
    count = 0
    num_track =0
    L = []
    roi_count = 0
    avg_roi = 0
    roi_prop = 0
    avg_prop = 0
    #f = open('knn-STIOU-trackroi-init-0.5-alpha-0.9-office.txt','a+')
    f = open('knn-STIOU-trackroi-sofa.txt','a+')
    
    while True:
        if i <= 200:
            i = i+1
            ok, frame = vid.read()
            #if i == 1:
                #g_w,g_h,_ = frame.shape
            image = Image.fromarray(frame)
            det_box = yolo.detect_image(image)
            x = det_box[0][0]
            y = det_box[0][1]
            w = det_box[0][2]
            h = det_box[0][3]
            det_bboxes=(x,y,w,h)
            det_bboxes = tuple(det_bboxes)
            detect_num = detect_num + 1
            if i == 1:
                bboxes = det_bboxes
                stracker = yolo.single_tracker(tracker_type)
                ok = stracker.init(frame, bboxes)
            ok, bboxes = stracker.update(frame)
            rect1 = (det_bboxes[0], det_bboxes[1], det_bboxes[0]+det_bboxes[2], det_bboxes[1]+det_bboxes[3])
            width = det_bboxes[2]
            height = det_bboxes[3]
            det_roi = width * height       
            rect2 = (bboxes[0], bboxes[1], bboxes[0]+bboxes[2], bboxes[1]+bboxes[3])
            width = bboxes[2]
            height = bboxes[3]
            track_roi =  width * height
            iou = compute_iou(rect1, rect2)
       
            if iou >= ST:
                num_track = num_track+1
                ST = 0.9*ST + 0.1*iou
       
                roi_count += track_roi
                #roi_prop += (track_roi / (g_w * g_h))
                count = 0
       
            else:
                if num_track > 0:
                    avg_roi = roi_count / num_track
                    #avg_prop = roi_prop / num_track
                    #L.append([num_track,avg_roi])
                    # tmp_L = [str(avg_prop),str(num_track)]
                    # f.write(str(tmp_L)+"\n")
                    f.write(str(avg_roi)+"\t"+str(num_track)+"\n")
                bboxes = det_bboxes
                stracker = yolo.single_tracker(tracker_type)
                ok = stracker.init(frame, bboxes)
                ok, bboxes = stracker.update(frame)
                num_track = 0
                roi_count = 0
                # roi_prop = 0
                ST1 = 0.5*ST
                ST2 = iou
                ST3 = 0.5
            
                ST_List = [ST1,ST2,ST3]
                #ST_List = [ST1,ST3]
                ST_List.sort()
                ST_List = ST_List[::-1]
                if count < 3:
                    ST = ST_List[count]
                else:
                    ST = ST_List[2]
                count += 1
       
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
       
       
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                a = a+1
       
                sum_fps = int(curr_fps) + sum_fps
                #print('sum_fps', sum_fps)
                #print('a ', a)
                curr_fps = 0
                avg_fps = sum_fps/a
       
                #print('avg-fps', avg_fps)
       
       
            # Draw bounding box
            if ok:
                # Tracking success
                #print('track_bboxes',type(bboxes),bboxes,det_bboxes)
                p1 = (int(bboxes[0]), int(bboxes[1]))
                p2 = (int(bboxes[0] + bboxes[2]), int(bboxes[1] + bboxes[3]))
                cv2.rectangle(frame, p1, p2, (0,0,255), 2, 1)
       
                p11 = (int(det_bboxes[0]), int(det_bboxes[1]))
                p22 = (int(det_bboxes[0] + det_bboxes[2]), int(det_bboxes[1] + det_bboxes[3]))
                #p11 = (int(det_bboxes[0][0]), int(det_bboxes[0][1]))
                #p22 = (int(det_bboxes[0][0] + det_bboxes[0][2]), int(det_bboxes[0][1] + det_bboxes[0][3]))
                cv2.rectangle(frame, p11, p22, (0,255,0), 2, 1)
       
       
            else :
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
       
                # Display tracker type on frame
            cv2.putText(frame, "Yolov3+"+tracker_type+str(i), (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
       
            #图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
            cv2.putText(frame, text=fps, org=(3, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.75, color=(255, 0, 0), thickness=2)
       
                # Display result
       
            #cv2.imwrite('./det-track/'+str(i)+'.jpg', frame)
            cv2.imshow("Tracking", frame)
            #if writeVideo_flag:
            #    out.write(frame)
            #    cv2.imwrite('./yolo-CSRT-picture/'+str(i)+'.jpg', frame)
            # Exit if ESC pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #print('detect_num', detect_num)
        else:
             break
        #del stracker 
            #print("len",len)
        
    vid.release()
    f.close()
    #if writeVideo_flag:
        #out.release()
        #list_file.close()
    yolo.close_session()
    cv2.destroyAllWindows()
		
		

		
if len(sys.argv) < 2:
    print("Usage: $ python {0} [video_path] [output_path(optional)]", sys.argv[0])
    exit()

if __name__ == '__main__':
    video_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
        detect_video(YOLO(), video_path, output_path)
    else:
        detect_video(YOLO(), video_path)
