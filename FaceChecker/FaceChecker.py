
from FaceChecker.config import *
import cv2
class FaceChecker:
    
    def __init__(self,head_pose_estimator,face_detector,noise_estimator):
        
        self.head_pose_estimator = head_pose_estimator
        self.face_detector = face_detector
        self.noise_estimator = noise_estimator
        
    def _caculate_noise(self,frame):

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

        signal = self.noise_estimator._caculate_noise_image(frame)

        return signal

    def test_on_video(self,video_path):
        
        cap = cv2.VideoCapture(video_path)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        prepare_img = cv2.imread('FaceChecker/portrait-photography.jpg')

        prepare_img = cv2.cvtColor(prepare_img,cv2.COLOR_BGR2RGB)

        results = self._run_on(prepare_img)

        print(results)

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            img_copy = frame.copy()

            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

            result = self.runOn(img_copy)
            
            if result is not None:
               
                for info in result:
                      x,y,w,h = info['coordinates']
                      cv2.rectangle(frame,(x-w,y-h),(x+w,y+h),(0,0,255),4)
                      yaw,pitch,roll = info['direction'][0]
                      text = 'yaw:{0:.2f}, pitch;{1:.2f}, roll: {2:.2f}'.format(yaw,pitch,roll)
                      cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), lineType=cv2.LINE_AA)

                cv2.imshow('result',frame)
            else:
                cv2.imshow('result',frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
         
    
    def runOn(self,frame):
        
        signal = self._caculate_noise(frame)

        print(signal)
        
        if signal >= NOISE_THRESHOLD:
           
            return None
          
        else:
          
            faces = self.face_detector.detect_faces(frame)
            
            checked_faces = self.face_detector.check_faces(faces)
            
            if len(checked_faces) < 1:
                
                return None
            
            else:
                
                checked_head_pose_faces = self.head_pose_estimator.getDirectionFromFaces(checked_faces)
                
                if len(checked_head_pose_faces) < 1:
                    
                    return None
                else:
                    
                    return checked_head_pose_faces