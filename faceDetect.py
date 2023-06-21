import cv2


def detectFaces(img):
    cascadePath="haarcascade_frontalface_default.xml"
    faceCascade=cv2.CascadeClassifier(cascadePath)
    grayFace=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(grayFace,scaleFactor=1.2,minNeighbors=5,minSize = (30,24))
    if(len(faces)==0):
        return None,None
    (x,y,w,h)=faces[0]
    return grayFace[y:y+w,x:x+h], faces[0]