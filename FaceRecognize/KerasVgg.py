from FaceRecognize.keras_vggface.vggface import VGGFace
import cv2
import numpy as np
import time
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
class FaceRecognize:

    def __init__(self,face_alignment,face_checker,input_shape=(224,224,3),backbone='resnet50',data_dir='FaceRecognize/data'):

        self.face_checker = face_checker
        self.input_shape = input_shape
        self.vggface = VGGFace(model=backbone,input_shape=input_shape,include_top=False)
        self.knn = KNeighborsClassifier()
        self.data_dir = data_dir
        self.face_alignment = face_alignment

        self.person_names = {
            'ben_afflek':0,
            'elton_john':1,
            'jerry_seinfeld':2,
            'madonna':3,
            'mindy_kaling':4
        }


        if not os.path.exists("FaceRecognize/embedding/embeddings.pkl"):
            self.preComputeEmbeddingData(self.data_dir)

        with open("FaceRecognize/embedding/embeddings.pkl",'rb') as f:
            self.embeddings = pickle.load(f)

        with open("FaceRecognize/embedding/labels.pkl", 'rb') as f:
            self.labels = pickle.load(f)

        self.embeddings = self.embeddings.reshape(-1,2048)

        self.knn.fit(self.embeddings,self.labels)

    def preComputeEmbeddingData(self,data_dir):

        print('in here')

        if not os.path.exists(data_dir):
            raise FileNotFoundError("Database Not Found")

        celeb_names = os.listdir(data_dir)
        cls = 0
        list_images = []
        list_labels = []
        for name in celeb_names:
            images_dir = os.path.join(data_dir,name)
            list_name_images = os.listdir(images_dir)
            cls = self.person_names[name]
            for image_name in list_name_images:
                image_path = os.path.join(images_dir,image_name)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                face = cv2.resize(image,self.input_shape[:2])
                list_images.append(face)
                list_labels.append(cls)

        list_images = np.array(list_images)
        list_labels = np.array(list_labels)

        list_embbeddings = self.vggface.predict(list_images)

        print(list_embbeddings.shape)

        with open("FaceRecognize/embedding/embeddings.pkl","wb") as f:
            pickle.dump(list_embbeddings,f)

        with open("FaceRecognize/embedding/labels.pkl", "wb") as f:
            pickle.dump(list_labels, f)

    def reconizeFace(self,face):
        face = cv2.resize(face,self.input_shape[:2])
        face = np.expand_dims(face,axis=0)
        embedding_face = self.vggface.predict(face)
        embedding_face = embedding_face.reshape(-1,2048)
        label = self.knn.predict(embedding_face)
        label = label[0]

        name = [k for k,v in self.person_names.items() if v == label]

        print(name)

    def test_on_image(self,image):

        result = self.face_checker.runOn(image)

        print(result)

        if result is not None:

            x,y,w,h = result[0]['coordinates']

            face = image[y-h:y+h,x-w:x+w]

            face_aligned = self.face_alignment.run(face)

            if face_aligned is not None:

                print('face aligned shape:',face_aligned.shape)

                self.reconizeFace(face_aligned)


if __name__ == '__main__':

     face = FaceRecognize()
    # face.preComputeEmbeddingData("FaceRecognize/data")
    # img = cv2.imread("FaceRecognize/data/ben_afflek/httpcsvkmeuaeccjpg.jpg")
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img,(224,224))
    # face.reconizeFace(img)




