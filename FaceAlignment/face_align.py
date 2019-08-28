from PIL import Image
from FaceAlignment.detector import FaceDetectAlign
from FaceAlignment.align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np
import cv2
import matplotlib.pyplot as plt

class FaceAlignment:

    def __init__(self):
        self.FaceDetectorAlign = FaceDetectAlign()

    def run(self,face):

        crop_size = 112
        scale = crop_size / 112.
        reference = get_reference_facial_points(default_square=True) * scale

        face = Image.fromarray(face)

        _, landmarks = self.FaceDetectorAlign.runOn(face)

        img_warped = None

        if len(landmarks) > 0:  # If the landmarks cannot be detected, the img will be discarded
            facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
            warped_face = warp_and_crop_face(np.array(face), facial5points, reference, crop_size=(crop_size, crop_size))
            img_warped = warped_face

        return img_warped

# def align_face(face):
#     crop_size = 112
#     scale = crop_size / 112.
#     reference = get_reference_facial_points(default_square = True) * scale
#
#     face = Image.fromarray(face)
#
#     _, landmarks = detect_faces(face)
#
#     if len(landmarks) > 0: # If the landmarks cannot be detected, the img will be discarded
#         facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
#         warped_face = warp_and_crop_face(np.array(face), facial5points, reference, crop_size=(crop_size, crop_size))
#         img_warped = Image.fromarray(warped_face)
#
#     return img_warped

if __name__ == '__main__':

    img = cv2.imread("E:\Source\PyFace\FaceChecker\portrait-photography.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(112,112))
    # face = align_face(img)
    # plt.imshow(face)
    # plt.show()