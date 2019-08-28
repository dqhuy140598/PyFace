from FaceChecker.FaceChecker import FaceChecker
from FaceChecker.HeadPoseEstimator.Headpose import HeadposeEstimation
from FaceChecker.config import FINAL_WEIGHTS_PATH,FACE_DETECTOR_MODEL_PATH
from FaceChecker.FaceDetector.FaceDetector import FaceDetector
from FaceChecker.NoiseEstimator.NoiseEstimator import NoiseEstimator
from FaceAlignment.face_align import FaceAlignment
import cv2
import matplotlib.pyplot as plt
from FaceRecognize.KerasVgg import FaceRecognize

if __name__ == '__main__':

    face_alignment = FaceAlignment()
    head_pose_estimator = HeadposeEstimation(FINAL_WEIGHTS_PATH)

    face_detector = FaceDetector(model_path=FACE_DETECTOR_MODEL_PATH)
    noise_estimator = NoiseEstimator()

    face_checker = FaceChecker(head_pose_estimator, face_detector, noise_estimator)

    face_recognize = FaceRecognize(face_alignment,face_checker)

    image = cv2.imread("AN2I5CXEDMZYPCD6YVAKOZLIUA.jpg")

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    face_recognize.test_on_image(image)


