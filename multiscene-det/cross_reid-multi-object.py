# -*- coding: utf-8 -*-
"""
多场景-多目标检测
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/tiny-yolo.h5',
        "anchors_path": 'model_data/tiny_yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        
        # "model_path": 'model_data/trained_weights.h5',
        # "anchors_path": 'model_data/yolo_anchors.txt',
        # "classes_path": 'model_data/ROI.txt',
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

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

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

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if predicted_class !='person':
                continue
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            #draw.rectangle(
                #[tuple(text_origin), tuple(text_origin + label_size)],
                #fill=self.colors[c])
            #draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        #end = timer()
        #print(end - start)
        return image

    def close_session(self):
        self.sess.close()

def detect_video(YOLO(), video_path1, video_path2, video_path3):
    import cv2
    vid1 = cv2.VideoCapture(video_path1)
    vid2 = cv2.VideoCapture(video_path2)
    vid3 = cv2.VideoCapture(video_path3)
    #vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    
    # fps = 24 #保存视频的FPS，可以适当调整 
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG') #可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg *'MJPG'
    # isOutput = True if output_path != "" else False
    # if isOutput:
        # out = cv2.VideoWriter(output_path,fourcc,fps,(1920,1080))#最后一个是保存图片的尺寸 
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    a = 1
    i = 0
    detect_num = 0
    sum_fps = 0
    #print('type-sum_fps', type(sum_fps))
    avg_fps = 0
    
    while True:
        
        ok1, frame1 = vid1.read()
        ok2, frame2 = vid2.read()
        ok3, frame3 = vid3.read()
        if not ok1 and not ok2 and not ok3: 
            print('avg-fps', avg_fps)
            break
        if ok1:
            image1 = Image.fromarray(frame1)
            image1 = yolo.detect_image(image1)
            result1 = np.asarray(image1)
        if ok2:
            image2 = Image.fromarry(frame2)
            image2 = yolo.detect_image(image2)
            result2 = np.asarray(image2)
        if ok3:
            image3 = Image.fromarray(frame3)
            image3 = yolo.detect_image(image3)
            result3 = np.asarray(image3)
            
        i = i+1 
            
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
                #print('avg-fps', avg_fps)
            cv2.putText(result, "Yolov3", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
            cv2.putText(result, text=fps, org=(3, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.75, color=(255, 0, 0), thickness=2)
            
            cv2.namedWindow("result", 0)
            cv2.imshow("result", result)
            #cv2.imwrite('./result/0/'+str(c).zfill(8)+'.jpg', result)
            # if isOutput:
                # out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('avg_fps', avg_fps)
                break

    yolo.close_session()

if __name__ == '__main__':
    # video_path1 = "../data/Reid/shulin-0.mp4"
    # video_path2 = "../data/Reid/shulin-1.mp4"
    # video_path3 = "../data/Reid/shulin-2.mp4"
    # video_path1 = "../data/5/output_0.mp4"
    # video_path2 = "../data/5/output_1.mp4"
    # video_path3 = "../data/5/output_2.mp4"
    video_path1 = 'rtsp://admin:jdh123456@10.11.99.226:554/H.264/ch1/sub/av_stream'#海康
    video_path2 = 'rtsp://admin:jdh123456@10.11.99.226:554/H.264/ch1/sub/av_stream'#海康
    video_path3 = 'rtsp://admin:jdh123456@10.11.99.226:554/H.264/ch1/sub/av_stream'#海康
    detect_video(YOLO(), video_path1, video_path2, video_path3)