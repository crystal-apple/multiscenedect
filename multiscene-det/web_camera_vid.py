import os 
import cv2 
#path = 'rtsp://admin:jdh123456@10.11.99.225:554/H.264/ch1/sub/av_stream'#海康
path = '../data/output_0.mp4'
cap = cv2.VideoCapture(path)

fps = 24 #保存视频的FPS，可以适当调整 #可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg *'MJPG'
fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
out = cv2.VideoWriter('/web_camera_result/camera1.avi',fourcc,fps,(640,480))#最后一个是保存图片的尺寸 
#按顺序循环读取图片
while True:
    ret, frame = cap.read()
    
    #out.write(frame)
    
    #cv2.namedWindow('frame')
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()

