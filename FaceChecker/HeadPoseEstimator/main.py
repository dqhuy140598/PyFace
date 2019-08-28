from FaceChecker.HeadPoseEstimator.Headpose import HeadposeEstimation
import cv2
import matplotlib.pyplot as plt
fsanet_model_path = ['/content/Headpose/face_direction/weights/fsanet_capsule_3_16_2_21_5.h5',
                    '/content/Headpose/face_direction/weights/fsanet_var_capsule_3_16_2_21_5.h5',
                    '/content/Headpose/face_direction/weights/fsanet_noS_capsule_3_16_2_192_5.h5']
temp = HeadposeEstimation(fsanet_model_path)

video_path = '/content/Tom Cruise Brings Les Grossman To #ConanCon - CONAN on TBS.mp4'

temp.test_on_video(video_path)