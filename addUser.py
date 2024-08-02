import cv2
import os
import faceDetect as fd
import numpy as np
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
data_path='training-data'
path='usernames/users.txt'
fr=open(path,'r')
str1=fr.read().split('/')
id=int(len(str1))
id=id-1
name=input('Enter user name:')
fp=open(path,'a')
fp.write(name+'/')

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480) 

print("\n [INFO] Initializing face capture. Look at the camera and wait ...")
count = 0

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(30,24))
    for (x,y,w,h) in faces:
        #cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("training-data/User." + str(id) + '.' + str(count) + ".jpg", img)
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 60: # Take 60 face sample and stop video
        break

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('trainer/trainer.yml')

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
faces,ids = getImagesAndLabels(data_path)
recognizer.update(faces, np.array(ids))

recognizer.write('trainer/trainer.yml')
print('new user face trained')
images=os.listdir(data_path)
for image in images:
    os.remove(data_path+'/'+image)
print('new user added')
