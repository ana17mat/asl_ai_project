from cv2 import imread
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import random

import numpy as np
from scipy.spatial import distance
import pickle


import pyttsx3

engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("voice", "com.apple.speech.synthesis.voice.moira")
# engine.say('hello my name is ana')
# engine.runAndWait()


# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720)  # height


# Create HandDetector object
detector = HandDetector(detectionCon=0.8, maxHands=2)


preds = [" "]

dict_asl = pickle.load(open("dictletras_a_to_t.pkl", "rb"))

positions = []
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # flip image in axis 1
    # detect the hand in each frame
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]["lmList"]
        pos = [l[0:2] for l in lmList]
        originwrist = [[pos[0][0] - l[0], pos[0][1] - l[1]] for l in pos]
        positions.append(pos)
    cv2.imshow("ASL", img)
    key = cv2.waitKey(1)  # 1 ms delay

    if not hands and preds[-1] != " ":
        preds.append(" ")
        print(" ")

    # pred with space key
    if key == 32:
        hpred = originwrist
        hmin = dict_asl["a"][0]
        dmin = np.mean([distance.euclidean(a, b) for a, b in zip(hpred, hmin)])
        letramin = "a"
        for letra in list(dict_asl.keys()):
            for h in dict_asl[letra]:
                d = np.mean([distance.euclidean(a, b) for a, b in zip(hpred, h)])
                if d < dmin:
                    dmin = d
                    letramin = letra
        print(letramin)
        preds.append(letramin)

    if key == ord("s"):
        preds_str = "".join(preds).strip()
        print(preds_str)
        engine.say(preds_str)
        engine.runAndWait()
        preds = []

    # CLOSE WITH ESC KEY
    if key == 27:
        break
