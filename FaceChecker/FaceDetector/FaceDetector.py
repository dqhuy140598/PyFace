

from keras.models import load_model
import numpy as np
import sys
import cv2
import os
from keras import backend as K
import time
from FaceChecker.FaceDetector.boundingbox import interpret_output_yolov2, crop
from FaceChecker.FaceDetector.config import IMAGE_SIZE_THESHOLD
os.environ['KERAS_BACKEND'] = 'tensorflow'

class FaceDetector:
    
    def __init__(self, model_path):
      self.detector = load_model(model_path)
   
    """
    def detect_faces(self,frame):
        
        t = time.time()

        frame_resized = cv2.resize(frame, (416,416))/255.0
        
        frame_resized = np.expand_dims(frame_resized, axis=0)
        
        predict = self.detector.predict(frame_resized)[0]
        
        results = interpret_output_yolov2(predict, np.shape(frame)[1], np.shape(frame)[0])
        
        list_faces = []
        
        for i in range(len(results)):
          
            if results[i][5] >= 0.5 and results[i][0] == 'face':
            
                
                #display detected face
                x = int(results[i][1])
                y = int(results[i][2])
                w = int(results[i][3])//2
                h = int(results[i][4])//2
                
                xmin, xmax, ymin, ymax = crop(
                    x, y, w, h, 1.4, np.shape(frame)[1], np.shape(frame)[0])
                
                print(str(i)+'--  '+str(xmin)+'-'+str(xmax)+'-'+str(ymin)+'-'+str(ymax))
                
                
                list_faces.append(frame[ymin:ymax, xmin:xmax])

        print('detect face time:{0:.4f}'.format(time.time()-t))

        return list_faces
        
        """
    
    def detect_faces(self,frame):
        print(np.shape(frame))
        t = time.time()

        frame_resized = cv2.resize(frame, (416,416))/255.0

        frame_resized = np.expand_dims(frame_resized, axis=0)

        print(np.shape(frame_resized))

        predict = self.detector.predict(frame_resized)[0]

        results = interpret_output_yolov2(predict, np.shape(frame)[1], np.shape(frame)[0])

        list_faces = []

        print(len(results))

        for i in range(len(results)):

            if results[i][5] >= 0.7 and results[i][0] == 'face':
                
                info = {}

                #display detected face
                x = int(results[i][1])
                y = int(results[i][2])
                w = int(results[i][3])//2
                h = int(results[i][4])//2

                xmin, xmax, ymin, ymax = crop(
                    x, y, w, h, 1.4, np.shape(frame)[1], np.shape(frame)[0])

                print(str(i)+'--  '+str(xmin)+'-'+str(xmax)+'-'+str(ymin)+'-'+str(ymax) )
                
                info['coordinates'] = [x,y,w,h]
                
                info['padding_face'] = frame[ymin:ymax, xmin:xmax]
                
                list_faces.append(info)

        print('detect face time:{0:.4f}'.format(time.time()-t))

        return list_faces
        
    def check_faces(self,list_faces):

        checked_faces = [face for face in list_faces if face['padding_face'].shape[0] >=IMAGE_SIZE_THESHOLD and face['padding_face'].shape[1] >=IMAGE_SIZE_THESHOLD]
        
        return checked_faces
      
    def test_on_video(self,video_path):
        
        cap = cv2.VideoCapture(video_path)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
  
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            list_faces = self.detect_faces(frame)
            
            result = self.check_faces(list_faces)
            
            if len(result) > 0:
               
                for info in result:
                      x,y,w,h = info['coordinates']
                      cv2.rectangle(frame,(x-w,y-h),(x+w,y+h),(0,0,255),4)
                      out.write(frame)
            else:
                out.write(frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        
        
      

if __name__=='__main__':
    
    face_detector = FaceDetector(model_path='/content/module2/FaceDetector/pretrain/yolov2_tiny-face.h5')
    
    video_path = '/content/module2/video.mp4'
    
    face_detector.test_on_video(video_path)
    

    
        