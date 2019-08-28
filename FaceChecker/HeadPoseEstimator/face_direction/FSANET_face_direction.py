import os
import cv2
import sys
import numpy as np
from math import cos, sin
from moviepy.editor import *
def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 80):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    Pit = "pit: "+str(round(pitch,6))
    Yaw = "yaw: "+str(round(yaw,6))
    Roll = "roll: "+str(round(roll,6))
    #cv2.putText(img, text=Pit, org=(0, 45), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #                    fontScale=0.50, color=(0, 0, 255), thickness=1)
    #cv2.putText(img, text=Yaw, org=(0, 75), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #                    fontScale=0.50, color=(0, 255, 0), thickness=1)
    #cv2.putText(img, text=Roll, org=(0, 105), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #                    fontScale=0.50, color=(255, 0, 0), thickness=1)                                        
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img
    
def draw_direction_results(results,input_img,faces,ad,img_size,img_w,img_h,model):
  
    if len(results) > 0:
        direction = []      
        for i, d in enumerate(results):
            x1,y1,w,h = d[0:4]
            x2,y2 = x1+w,y1+h
            x = (x1+x2)//2
            y = (y1+y2)//2
            print(x,y,w,h)
            xw1 = max(int(x - ad * w), 0)
            yw1 = max(int(y - ad * h), 0)
            xw2 = min(int(x + ad * w), img_w - 1)
            yw2 = min(int(y + ad * h), img_h - 1)
            #print(str(xw1)+' '+str(yw1)+' '+str(xw2)+' '+str(yw2))
            
            print(input_img[yw1:yw2+1, xw1:xw2+1, :].shape)
            
            faces[i,:,:,:] = cv2.resize(input_img[yw1:yw2+1, xw1:xw2+1, :], (64, 64))
            faces[i,:,:,:] = cv2.normalize(faces[i,:,:,:], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)        
            
            face = np.expand_dims(faces[i,:,:,:], axis=0)
            p_result = model.predict(face)
            
            face = face.squeeze()
            img = draw_axis(input_img[yw1:yw2+1, xw1:xw2+1, :], p_result[0][0], p_result[0][1], p_result[0][2])
            #print(p_result.shape)
            input_img[yw1:yw2+1, xw1:xw2+1, :] = img
            
            direction.append(p_result)

    return direction