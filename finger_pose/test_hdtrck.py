from cv2 import imread
#import cvzone
import cv2
from matplotlib.pyplot import draw
import numpy as np
from cvzone.HandTrackingModule import HandDetector
#import random
import getch
#import time

from scipy.spatial import distance
import pickle

from torch import flip
from os import system

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720)  # height

# Create HandDetector object
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # flip image in axis 1
    hands, img = detector.findHands(img, flipType=False)
    cv2.imshow("ASL", img)
    key = cv2.waitKey(1)  # 1 ms delay
    # CLOSE WITH ESC KEY
    if key == 27:
        break
