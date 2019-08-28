
import math
import numpy as np
from scipy.signal import convolve2d
import cv2
class NoiseEstimator:

        def __init__(self):
               pass

        def _caculate_noise_image(self,frame):
            H, W = frame.shape

            M = [[1, -2, 1],
                 [-2, 4, -2],
                 [1, -2, 1]]

            sigma = np.sum(np.sum(np.absolute(convolve2d(frame, M))))
            sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W - 2) * (H - 2))

            return sigma

if __name__=="__main__":

        image_path = 'IRG1L.png'
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        noise_estimator = NoiseEstimator()
        signal = noise_estimator._caculate_noise_image(image)
        print(signal)


