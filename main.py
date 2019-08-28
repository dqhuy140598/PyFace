from FaceChecker.FaceChecker import FaceChecker
from FaceChecker.HeadPoseEstimator.Headpose import HeadposeEstimation
from FaceChecker.config import FINAL_WEIGHTS_PATH,FACE_DETECTOR_MODEL_PATH
from FaceChecker.FaceDetector.FaceDetector import FaceDetector
from FaceChecker.NoiseEstimator.NoiseEstimator import NoiseEstimator


head_pose_estimator = HeadposeEstimation(FINAL_WEIGHTS_PATH)
face_detector = FaceDetector(model_path=FACE_DETECTOR_MODEL_PATH)
noise_estimator = NoiseEstimator()

face_checker = FaceChecker(head_pose_estimator,face_detector,noise_estimator)

video_path = 'FaceChecker/test.mp4'

face_checker.test_on_video(video_path)