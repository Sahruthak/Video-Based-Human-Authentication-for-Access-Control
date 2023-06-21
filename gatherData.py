import cv2
import os

class gather():
    def __init__(self):
        self.__init__=self


def gather():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    names=[]
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    n=int(input('\nEnter no of users:'))
    i=0
    while(i<n):

        name=input(('\nEnter username:'))
        names.append(name)

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
                cv2.imwrite("training-data/User." + str(i) + '.' + str(count) + ".jpg", img)
                cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 60: # Take 60 face sample and stop video
                break
        i=i+1
        if i<n:
            print("next user:")
        cv2.waitKey(5000)

    path='usernames/users.txt'
    fp=open(path,'a')
    for na in names:
        #print(na)
        fp.write(na+'/')

    cam.release()
    cv2.destroyAllWindows()

gather()
