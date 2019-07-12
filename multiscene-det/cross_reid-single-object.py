# -*- coding: utf-8 -*-
"""
多场景-单目标 检测 
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


# global tracker
# global stracker

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/tiny-yolo.h5',
        "anchors_path": 'model_data/tiny_yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        # self.tracker = self.single_tracker(self,tracker_type)
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

        # print('{} model, anchors, and classes loaded.'.format(model_path))

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
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        # start = timer()
        # return_boxes = []
        global return_boxes

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        return_boxes = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if predicted_class != 'person':
                continue

            box = out_boxes[i]

            x = int(box[1])
            y = int(box[0])
            w = int(box[3] - box[1])
            h = int(box[2] - box[0])
            if x < 0:
                w = w + x
                x = 0
            if y < 0:
                h = h + y
                y = 0

            bboxx = (x, y, w, h)
            return_boxes.append(bboxx)
        return return_boxes

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path1, video_path2, video_path3):
    # global return_boxes
    global width
    global height
    import cv2
    global tracker1
    global tracker2
    global tracker3
    vid1 = cv2.VideoCapture(video_path1)
    vid2 = cv2.VideoCapture(video_path2)
    vid3 = cv2.VideoCapture(video_path3)
    # fps = 24 #保存视频的FPS，可以适当调整 
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG') #可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg *'MJPG'
    # out = cv2.VideoWriter(output_path,fourcc,fps,(272,480))#最后一个是保存图片的尺寸 

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    detect_num = 0
    sum_fps = 0
    # print('type-sum_fps', type(sum_fps))
    avg_fps = 0
    # fps = 0.0q

    x = 0
    y = 0
    z = 0

    a = 1
    f = open('cameraID-personID-coord.txt', 'w')
    while True:

        ok1, frame1 = vid1.read()
        ok2, frame2 = vid2.read()
        ok3, frame3 = vid3.read()
        
        if x % 5 == 0 and ok1:
            #print('1111111111111', x)
            image1 = Image.fromarray(frame1)
            bboxes1 = yolo.detect_image(image1)
            if len(bboxes1) > 0:
                bboxes1 = bboxes1[0]
            else:
                # bboxes1 = (230, 189, 203, 279)
                bboxes1 = (1, 1, 2, 2)
            #print('bboxes1', bboxes1)
            tracker1 = cv2.TrackerCSRT_create()
            
            ret1 = tracker1.init(frame1, bboxes1)
        ret1, bboxe1 = tracker1.update(frame1)
        x += 1

        if y % 5 == 0:
            image2 = Image.fromarray(frame2)
            bboxes2 = yolo.detect_image(image2)
            if len(bboxes2) > 0:
                bboxes2 = bboxes2[0]
            else:
                bboxes2 = (1, 1, 2, 2)
            tracker2 = cv2.TrackerCSRT_create()
            ret2 = tracker2.init(frame2, bboxes2)
        ret2, bboxe2 = tracker2.update(frame2)
        y += 1

        if z % 5 == 0:
            image3 = Image.fromarray(frame3)
            bboxes3 = yolo.detect_image(image3)
            if len(bboxes3):
                bboxes3 = bboxes3[0]
            else:
                bboxes3 = (1, 1, 2, 2)
            tracker3 = cv2.TrackerCSRT_create()
            ret3 = tracker3.init(frame3, bboxes3)
        ret3, bboxe3 = tracker3.update(frame3)
        z += 1
        '''
        ok1, frame1 = vid1.read()
        ok2, frame2 = vid2.read()
        ok3, frame3 = vid3.read()

        if not ok1 and not ok2 and not ok3:
            print('avg_fps', avg_fps)
            break

        try:
            if x%5  == 0:
                
                image1 = Image.fromarray(frame1)
                
                bboxes1 = yolo.detect_image(image1)
                if len(bboxes1): 
                    bboxes1 = bboxes1[0]
                    stracker1 = cv2.TrackerCSRT_create()
                    ok1 = stracker1.init(frame1,bboxes1)
                    print(1111111111111)
                ret1,bboxe1 = stracker1.update(frame1)
                print(5555555555555)
        except:
            continue
          
        try:
            if y%5  == 0:
                image2 = Image.fromarray(frame2)
                bboxes2 = yolo.detect_image(image2)
                if len(bboxes2):
                    bboxes2 = bboxes2[0]
                    stracker2 = cv2.TrackerCSRT_create()
                    ok2 = stracker2.init(frame2,bboxes2)
                ret2,bboxe2 = stracker2.update(frame2)
        except:
            continue
            
        try:
            if z%5  == 0:
                image3 = Image.fromarray(frame3)
                bboxes3 = yolo.detect_image(image3)
                if len(bboxes3):
                    bboxes3 = bboxes3[0]
                    stracker3 = cv2.TrackerCSRT_create()
                    ok3 = stracker3.init(frame3,bboxes3)
                ret3,bboxe3 = stracker3.update(frame3)
        except:
            continue
        '''
        x += 1
        y += 1
        z += 1

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1

        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            a = a + 1

            sum_fps = int(curr_fps) + sum_fps
            curr_fps = 0
            avg_fps = sum_fps / a
        if ret1:
            f.write("camera-1" + "\t" + "personID:" + str(1) + "\t" + "time:" + str(
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + "\t" + "coord:[" + str(
                int(bboxe1[0])) + " " + str(int(bboxe1[1])) + " " + str(int(bboxe1[2])) + " " + str(
                int(bboxe1[3])) + "]\n")
            p1 = (int(bboxe1[0]), int(bboxe1[1]))
            p2 = (int(bboxe1[0] + bboxe1[2]), int(bboxe1[1] + bboxe1[3]))            
            times = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            if int(bboxe1[0]) > 100:
                print('camera-1   personID-1   time:', times,'coord:[',int(bboxe1[0]),int(bboxe1[1]),int(bboxe1[0] + bboxe1[2]), int(bboxe1[1] + bboxe1[3]),']\n')
            cv2.rectangle(frame1, p1, p2, (0, 255, 0), 2, 1)
            
        else:
            # Tracking failure
            cv2.putText(frame1, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        if ret2:
            f.write("camera-2" + "\t" + "personID:" + str(1) + "\t" + "time:" + str(
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + "\t" + "coord:[" + str(
                int(bboxe2[0])) + " " + str(int(bboxe2[1])) + " " + str(int(bboxe2[2])) + " " + str(
                int(bboxe2[3])) + "]\n")
            p1 = (int(bboxe2[0]), int(bboxe2[1]))
            p2 = (int(bboxe2[0] + bboxe2[2]), int(bboxe2[1] + bboxe2[3]))
            times = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            if int(bboxe2[0]) > 100:
                print('camera-2   personID-1   time:', times,'coord:[',int(bboxe2[0]),int(bboxe2[1]),int(bboxe2[0] + bboxe2[2]), int(bboxe2[1] + bboxe2[3]),']\n')
            cv2.rectangle(frame2, p1, p2, (0, 255, 0), 2, 1)

        else:
            # Tracking failure
            cv2.putText(frame2, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        if ret3:
            f.write("camera-3" + "\t" + "personID:" + str(1) + "\t" + "time:" + str(
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + "\t" + "coord:[" + str(
                int(bboxe3[0])) + " " + str(int(bboxe3[1])) + " " + str(int(bboxe3[2])) + " " + str(
                int(bboxe3[3])) + "]\n")
            p1 = (int(bboxe3[0]), int(bboxe3[1]))
            p2 = (int(bboxe3[0] + bboxe3[2]), int(bboxe3[1] + bboxe3[3]))
            times = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            if int(bboxe3[0]) > 100:
                print('camera-3   personID-1   time:', times,'coord:[',int(bboxe3[0]),int(bboxe3[1]),int(bboxe3[0] + bboxe3[2]), int(bboxe3[1] + bboxe3[3]),']\n')
            cv2.rectangle(frame3, p1, p2, (0, 255, 0), 2, 1)

        else:
            # Tracking failure
            cv2.putText(frame3, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # Display tracker type on frame
        cv2.putText(frame1, "YOLOv3+CSRT", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame2, "YOLOv3+CSRT", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame3, "YOLOv3+CSRT", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
        # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        cv2.putText(frame1, text=fps, org=(3, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75, color=(255, 0, 0), thickness=2)
        cv2.putText(frame2, text=fps, org=(3, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75, color=(255, 0, 0), thickness=2)
        cv2.putText(frame3, text=fps, org=(3, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75, color=(255, 0, 0), thickness=2)
        # cv2.imwrite('./evaluate1/yolo-csrt-ren-origin/'+str(i)+'.jpg', frame)
        cv2.namedWindow('frame1', 0)
        cv2.moveWindow("frame1",200,500)
        cv2.imshow("frame1", frame1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.namedWindow('frame2', 0)
        cv2.moveWindow("frame2",650,500)
        cv2.imshow("frame2", frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.namedWindow('frame3', 0)
        cv2.moveWindow("frame3",1100,500)
        cv2.imshow("frame3", frame3)
        # Exit if ESC pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # if isOutput:
        # out.write(frame)
    # del stracker
    vid1.release()
    vid2.release()
    vid3.release()

    # knn_result.close()
    # if writeVideo_flag:
    # out.release()

    # del tracker
    K.clear_session()
    yolo.close_session()
    cv2.destroyAllWindows()
    return 0


K.clear_session()

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
