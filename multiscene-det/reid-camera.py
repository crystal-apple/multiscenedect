# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import sys
import colorsys
import os
from timeit import default_timer as timer
import time
import numpy as np
from keras import backend as K
K.clear_session()
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import cv2
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model


#global tracker
#global stracker

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
        #self.tracker = self.single_tracker(self,tracker_type)
        global return_boxes 
        return_boxes = []
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
        
        #print('{} model, anchors, and classes loaded.'.format(model_path))

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
        #return_boxes = []
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
                del return_boxes[:]
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


    def close_session(self):
        self.sess.close()

def detect_video(yolo, scene, video_path, output_path=""):
    # global return_boxes
    global width
    global height
    import cv2
    global stracker
    vid = cv2.VideoCapture(video_path)
    # vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    fps = 24 #保存视频的FPS，可以适当调整 
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') #可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg *'MJPG'
    isOutput = True if output_path != "" else False
    if isOutput:
        out = cv2.VideoWriter(output_path,fourcc,fps,(272,480))#最后一个是保存图片的尺寸 
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    detect_num = 0
    sum_fps = 0
    # print('type-sum_fps', type(sum_fps))
    avg_fps = 0
    # fps = 0.0q

    i = 0
    a = 1
    f = open('cameraID-personID-coord.txt','a+')
    while True:
        ok, frame = vid.read()

                
        if not ok:
            print('avg_fps', avg_fps)
            break
        # flag = True
        # while flag:
            # ok, frame = vid.read()
            # image = Image.fromarray(frame)
            # bboxes = yolo.detect_image(image)
            # if len(bboxes) != 0:
                # flag = False
        try:
            if i%5  == 0:
                image = Image.fromarray(frame)
                bboxes = yolo.detect_image(image)
                if len(bboxes):
                    bboxes = bboxes[0]
                    stracker = cv2.TrackerCSRT_create()
                    ok = stracker.init(frame,bboxes)
                ret,bboxe = stracker.update(frame)
                #print('retttttttttttttt', ret)
        except:
            continue
        i += 1
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
            curr_fps = 0
            avg_fps = sum_fps/a
           
        if ret:
            #f.write("cameraID:"+str(scene)+"\t"+"personID:"+str(1)+"\t"+"time:"+str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))+"\t"+"coord:["+str(int(bboxes[0]))+" "+str(int(bboxes[1]))+" "+str(int(bboxes[2]))+" "+str(int(bboxes[3]))+"]\n")
            p1 = (int(bboxe[0]), int(bboxe[1]))
            p2 = (int(bboxe[0] + bboxe[2]), int(bboxe[1] + bboxe[3]))
            cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
            
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
            # Display tracker type on frame
        cv2.putText(frame, "YOLOv3+CSRT", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
    
        #图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        cv2.putText(frame, text=fps, org=(3, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75, color=(255, 0, 0), thickness=2)

        #cv2.imwrite('./evaluate1/yolo-csrt-ren-origin/'+str(i)+'.jpg', frame)
        cv2.imshow("Tracking", frame)
        # if isOutput:
            # out.write(frame)
        # Exit if ESC pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # del stracker
    vid.release()
    # knn_result.close()
    # if writeVideo_flag:
    # out.release()

    #del tracker
    K.clear_session()
    yolo.close_session()
    cv2.destroyAllWindows()
    return 0


K.clear_session()


if __name__ == '__main__':
    scenes = ["shulin-0", "shulin-1","shulin-2"]
    #scenes = ["CaVignal"]
    for scene in scenes:
        print(scene)
        video_path = "../data/Reid/" + str(scene) + ".mp4"
        output_path = "./result/" + str(scene) + ".avi"
        detect_video(YOLO(),scene, video_path, output_path)