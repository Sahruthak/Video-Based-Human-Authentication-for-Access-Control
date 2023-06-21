import cv2
import numpy as np
from PIL import Image
import os
import faceDetect as fd

# Path for face image database
path = 'training-data'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# function to get the images and label data
faces=[]
ids=[]
def getImagesAndLabels(data_path):
    faces=[]
    ids=[]
   
    img_names=os.listdir(data_path)
    for img_name in img_names:
        str=img_name.split('.')
        id=int(str[1])
        img_path=data_path+"/"+img_name
        image=cv2.imread(img_path)
        cv2.imshow("Training image...",cv2.resize(image,(320,240)))
        cv2.waitKey(100)

        face,rect =fd.detectFaces(image)
            #print(len(faces))
        if face is not None:
            faces.append(face)
            ids.append(id)
    return faces,ids
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

print(len(faces))
print(len(ids))


images=os.listdir(path)
for image in images:
    os.remove(path+'/'+image)
print("Training done...")