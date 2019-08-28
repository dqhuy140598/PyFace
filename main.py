from FaceChecker.FaceChecker import FaceChecker
from FaceChecker.HeadPoseEstimator.Headpose import HeadposeEstimation
from FaceChecker.config import FINAL_WEIGHTS_PATH,FACE_DETECTOR_MODEL_PATH
from FaceChecker.FaceDetector.FaceDetector import FaceDetector
from FaceChecker.NoiseEstimator.NoiseEstimator import NoiseEstimator
from FaceAlignment.face_align import FaceAlignment
import cv2
import matplotlib.pyplot as plt
# head_pose_estimator = HeadposeEstimation(FINAL_WEIGHTS_PATH)
# face_detector = FaceDetector(model_path=FACE_DETECTOR_MODEL_PATH)
# noise_estimator = NoiseEstimator()
#
# face_checker = FaceChecker(head_pose_estimator,face_detector,noise_estimator)
#
# video_path = 'FaceChecker/test.mp4'
#
# face_checker.test_on_video(video_path)

img = cv2.imread("FaceChecker/BZ7LGZSXCFHOXAN2XCTAM4QRPQ.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

face_alignment = FaceAlignment()

face = face_alignment.run(img)

plt.imshow(face)

plt.show()