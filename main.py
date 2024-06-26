import cv2
import numpy as np
import os 
import winsound

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names=[]
path='usernames/users.txt'
fp=open(path,'r')
s=fp.read().split('/')
for name in s:
    #print(name)
    names.append(name)
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480) 

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

fl=False

while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.05,
        minNeighbors = 5,
        minSize = (30,24),#before int(minW), int(minH)
       )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence < 100):
            name = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            name = "unknown"
            confidence = "  {0}%".format(round(confidence-100))
            fl=True
            
        
        cv2.putText(img, str(name), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        if fl==True:
            break
    cv2.imshow('camera',img) 
    if(fl==True):
        break
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
if(fl==True):
    while(True):
        duration = 300  # milliseconds
        freq = 440  # Hz
        winsound.Beep(freq, duration)
        k = cv2.waitKey(10) & 0xff
        if(k==ord('k')):
            break

cam.release()
cv2.destroyAllWindows()
