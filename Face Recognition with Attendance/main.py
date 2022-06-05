import cv2

import numpy as np
import face_recognition
import os
from datetime import datetime


path = "Training_images"
 
images = []
classNames = []
MyList = os.listdir(path)
print(MyList)
for cls in MyList:
    currentimage = cv2.imread(f'{path}/{cls}')
    images.append(currentimage)
    classNames.append(os.path.splitext(cls)[0])
    
print(classNames)


def findEncoding(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode_img = face_recognition.face_encodings(img)[0]
        encodeList.append(encode_img)
        
    return encodeList

knownImg_encodeList = findEncoding(images)

print("Encodding Complete")


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()

        # print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
                
                
cap =cv2.VideoCapture(0)
while True:
    
        success , img = cap.read()
        if success==True:
            imgResize = cv2.resize(img,(0,0) , None , 0.25, 0.25)
            imgResize_RGB = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB) 

            face_Currentframe = face_recognition.face_locations(imgResize_RGB)
            encode_Currentframe = face_recognition.face_encodings(imgResize_RGB , face_Currentframe)

            for encodeface , faceLoc in zip(encode_Currentframe,face_Currentframe):
                matches = face_recognition.compare_faces(knownImg_encodeList, encodeface)
                faceDis = face_recognition.face_distance(knownImg_encodeList, encodeface)
                    
                    # print(faceDis)
                    
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    # print(name)
                    y1,x2,y2,x1=faceLoc
                    y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    markAttendance(name)
        
            cv2.imshow('Webcam', img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                    break   
                 
        else:
            break
