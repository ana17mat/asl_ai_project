from cv2 import imread
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import random

import numpy as np
from scipy.spatial import distance
import pickle

dictletras={}

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720)  # height

# Create HandDetector object
detector = HandDetector(detectionCon=0.8, maxHands=2)

positions = []
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # flip image in axis 1
    # detect the hand in each frame
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        # get the landmark point of the index finger
        lmList = hands[0]['lmList']
        pos = [l[0:2] for l in lmList]
        originwrist = [[pos[0][0]-l[0], pos[0][1]-l[1]] for l in pos]
        positions.append(pos)
    cv2.imshow("ASL", img)
    key = cv2.waitKey(1)  # 1 ms delay



    if key in range(97,123):
        try:
            dictletras[chr(key)]+=[originwrist]
        except:
            dictletras[chr(key)]=[originwrist]



    # CLOSE WITH ESC KEY 
    if key == 27:

        print(dictletras)

        # create a binary pickle file 
        f = open("dictletras_sdfhgjhkj.pkl","wb")
        # write the python object (dict) to pickle file
        pickle.dump(dictletras,f)
        # close file
        f.close()
        break