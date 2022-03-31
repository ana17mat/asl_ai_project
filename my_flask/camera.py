import cv2
import threading
import time
import logging
from cvzone.HandTrackingModule import HandDetector
import pickle
from scipy.spatial import distance
import numpy as np
from os import system
import warnings
warnings.filterwarnings(action='ignore')

detector = HandDetector(detectionCon=0.8, maxHands=1)

preds = [" "]
knn_asl = pickle.load(
    open("nbs_dicts_models/knn_asl_162100_newo.p", "rb"))
#preds_str = "".join(preds).strip()
# knn_asl_161828.p knn_asl_162034.p
# 'knn_asl_162100_newo.p'
# rf_newo_171333.p
# knn_asl_171713_newo.p


minf = 4  # minimum frames in a row to make prediction
preds_m = [" "]*(minf+1)


logger = logging.getLogger(__name__)

thread = None


class Camera:
    def __init__(self, fps=20, video_source=0):
        logger.info(
            f"Initializing camera class with {fps} fps and video_source={video_source}"
        )
        self.fps = fps
        self.video_source = video_source
        self.camera = cv2.VideoCapture(self.video_source)
        # We want a max of 5s history to be stored, thats 5s*fps
        self.max_frames = 5 * self.fps
        self.frames = []
        self.isrunning = False
       

    def run(self):
        logging.debug("Preparing thread")
        global thread
        if thread is None:
            logging.debug("Creating thread")
            thread = threading.Thread(target=self._capture_loop, daemon=True)
            logger.debug("Starting thread")
            self.isrunning = True
            thread.start()
            logger.info("Thread started")

    def _capture_loop(self):
        dt = 1 / self.fps
        logger.debug("Observation started")
        while self.isrunning:
            v, im = self.camera.read()
            # added:
            im = cv2.flip(im, 1)
            hands, im = detector.findHands(im, flipType=False)

            preds_str = "".join(preds).strip().upper()

            cv2.putText(
                im,
                preds_str[-30:],
                # f"LETRA:{str('A')}",
                (30, 700),
                cv2.FONT_HERSHEY_DUPLEX,
                2,
                (172, 90, 255),
                2,
            )

            if hands:
                lmList = hands[0]["lmList"]
                pos = [l[0:2] for l in lmList]


                # ORIGINWRIST
                # originw = [[pos[0][0] - l[0], pos[0][1] - l[1]]
                #             for l in pos]


                # ORIGIN PREVIOUS LANDMARK
                originw = pos
                for i in range(20, 0, -1):
                    originw[i][0] = originw[i-1][0]-originw[i][0]
                    originw[i][1] = originw[i-1][1]-originw[i][1]
                originw[0] = [0, 0]

             
                hpred = [item for sublist in originw for item in sublist]

                letramin = knn_asl.predict([hpred])[0]
                preds_m.append(letramin)
          

                if all([preds_m[i] == preds_m[i+1] for i in range(-minf, -1)]+[preds_m[-minf-1] != preds_m[-minf]]):
                    preds.append(preds_m[-1])

   

            if not hands and preds[-1] != " ":
                to_say = "".join(preds).strip().split(' ')[-1]
                system('say '+to_say)
                preds.append(" ")
      

            if v:
                if len(self.frames) == self.max_frames:
                    self.frames = self.frames[1:]
                self.frames.append(im)

            time.sleep(dt)
        logger.info("Thread stopped successfully")

    def stop(self):
        logger.debug("Stopping thread")
        self.isrunning = False

    def get_frame(self, _bytes=True):

        if _bytes:
            img1 = cv2.resize(self.frames[-1], (1152, 648))
            img = cv2.imencode(".png", img1)[1].tobytes()
        else:
            img = self.frames[-1]
        return img
