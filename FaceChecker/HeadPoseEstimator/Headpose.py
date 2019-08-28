
import os
import cv2
import sys
import numpy as np
from math import cos, sin
from moviepy.editor import *
from moviepy.editor import *
from keras import backend as K
import argparse
import matplotlib.pyplot as plt
from FaceChecker.HeadPoseEstimator.face_direction.FSANET_model import FSA_net_Capsule, FSA_net_Var_Capsule,FSA_net_noS_Capsule
from FaceChecker.HeadPoseEstimator.face_direction.FSANET_face_direction import *
from keras.layers import Input,Average
from keras.models import Model
import time
from FaceChecker.HeadPoseEstimator.config import *

class HeadposeEstimation:

  def __init__(self,fsanet_model_path,option=None):

      if option is None:

         self.option = {
              'image_size' : 64,
              'num_capsule' : 3,
              'dim_capsule' : 16,
              'routings' : 2,
              'stage_num' : [3,3,3],
              'lambda_d' : 1,
              'num_classes' : 3,
              'image_size' : 64,
              'num_primcaps' : 7*3,
              'm_dim' : 5,
              'ad' : 0.6
         }
      else:
          self.option = option

      self.image_size = self.option['image_size']
      self.channels = 3

      fsanet_model_path = fsanet_model_path

      S_set = [self.option['num_capsule'], 
                    self.option['dim_capsule'], 
                    self.option['routings'], 
                    self.option['num_primcaps'], 
                    self.option['m_dim']
                   ]

      model1 = FSA_net_Capsule(self.option['image_size'], self.option['num_classes'], self.option['stage_num'], self.option['lambda_d'], S_set)()

      model2 = FSA_net_Var_Capsule(self.option['image_size'], self.option['num_classes'], self.option['stage_num'], self.option['lambda_d'], S_set)()

      self.option['num_primcaps'] = 8*8*3

      S_set = [self.option['num_capsule'], 
                    self.option['dim_capsule'], 
                    self.option['routings'], 
                    self.option['num_primcaps'], 
                    self.option['m_dim']
                   ]

      model3 = FSA_net_noS_Capsule(self.image_size, self.option['num_classes'], self.option['stage_num'], self.option['lambda_d'], S_set)()

      print('Loading models ...')

      weight_file1 = fsanet_model_path[0]
      model1.load_weights(weight_file1)
      print('Finished loading model 1.')

      weight_file2 = fsanet_model_path[1]
      model2.load_weights(weight_file2)
      print('Finished loading model 2.')

      weight_file3 = fsanet_model_path[2]
      model3.load_weights(weight_file3)
      print('Finished loading model 3.')

      inputs = Input(shape=(self.image_size,self.image_size,self.channels))

      x1 = model1(inputs) #1x1
      x2 = model2(inputs) #var
      x3 = model3(inputs) #w/o

      avg_model = Average()([x1,x2,x3])

      self.model = Model(inputs=inputs, outputs=avg_model)
  
  def checkHeadPose(self,info):
      
      print(info['direction'][0])
      
      if (info['direction'][0][0] >=-HEAD_POSE_THRESHOLD and info['direction'][0][0] <=HEAD_POSE_THRESHOLD) \
          and (info['direction'][0][1] >=-HEAD_POSE_THRESHOLD and info['direction'][0][1] <= HEAD_POSE_THRESHOLD) \
          and (info['direction'][0][2] >=-HEAD_POSE_THRESHOLD and info['direction'][0][2] <= HEAD_POSE_THRESHOLD):
          return True

      else:
          
          return False
    
  def getDirectionFromFaces(self,list_faces):

    t = time.time()

    if(len(list_faces)>0):

      results = []

      for data in list_faces:
        
        face = data['padding_face']
        
        x,y,w,h = data['coordinates']

        info = {}

        face_to_predict = cv2.resize(face, (64,64))

        face_to_predict = cv2.normalize(face_to_predict, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        face_to_predict  = np.expand_dims(face_to_predict, axis=0)

        p_result = self.model.predict(face_to_predict)

        img = draw_axis(face, p_result[0][0], p_result[0][1], p_result[0][2])

        info['direction'] = p_result
        
        info['coordinates'] = data['coordinates']
        
        if self.checkHeadPose(info):
             results.append(info)

        print('head pose estimation time:{0:.2f}'.format(time.time()-t))

    return results
  
"""  
  def getDirectionFromFaces(self,faces_coordinate,input_img):

      t = time.time()

      results = []  

      if len(faces_coordinate) > 0:

          print(len(faces_coordinate))    

          img_h,img_w,_ = input_img.shape

          faces = np.empty((len(faces_coordinate),self.image_size,self.image_size,self.channels))

          for i, d in enumerate(faces_coordinate):
              info = {}
              x1,y1,w,h = d[0:4]
              x2,y2 = x1+w,y1+h
              x = (x1+x2)//2
              y = (y1+y2)//2
              print(x,y,w,h)
              xw1 = max(int(x - self.option['ad'] * w), 0)
              yw1 = max(int(y - self.option['ad'] * h), 0)
              xw2 = min(int(x + self.option['ad'] * w), img_w - 1)
              yw2 = min(int(y + self.option['ad'] * h), img_h - 1)
              #print(str(xw1)+' '+str(yw1)+' '+str(xw2)+' '+str(yw2))

              print(input_img[yw1:yw2+1, xw1:xw2+1, :].shape)

              faces[i,:,:,:] = cv2.resize(input_img[yw1:yw2+1, xw1:xw2+1, :], (64, 64))
              faces[i,:,:,:] = cv2.normalize(faces[i,:,:,:], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)        

              face = np.expand_dims(faces[i,:,:,:], axis=0)
              p_result = self.model.predict(face)
              
              img = draw_axis(input_img[yw1:yw2+1, xw1:xw2+1, :], p_result[0][0], p_result[0][1], p_result[0][2])
              
              #print(p_result.shape)
              input_img[yw1:yw2+1, xw1:xw2+1, :] = img

              info['box'] = [x1,y1,w,h]
              info['direction'] = p_result
              info['image'] = input_img
              
              if checkHeadPose(info):
                   results.append(info)
                  
      print('head pose estimation time:{0:.2f}'.format(time.time()-t))

      return results
"""
      