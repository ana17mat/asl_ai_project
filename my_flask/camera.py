import cv2
import threading
import time
import logging
from cvzone.HandTrackingModule import HandDetector
import pickle
from scipy.spatial import distance
import numpy as np
import getch

detector = HandDetector(detectionCon=0.8, maxHands=2)
positions = []
preds = [" "]
dict_asl = pickle.load(
    open("/Users/anamatias/Desktop/FINAL_PROJECT/asl_project/dictletras_a_to_o.pkl", "rb")
)


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

            if hands:
                lmList = hands[0]["lmList"]
                pos = [l[0:2] for l in lmList]
                originwrist = [[pos[0][0] - l[0], pos[0][1] - l[1]] for l in pos]
                positions.append(pos)

                # key = getch.getch()  # cv2.waitKey(1)  # 1 ms delay
                # print("KEY:", key)
                # pred with space key
                # if key == " ":
                hpred = originwrist
                hmin = dict_asl["a"][0]
                dmin = np.mean([distance.euclidean(a, b) for a, b in zip(hpred, hmin)])
                letramin = "a"
                for letra in list(dict_asl.keys()):
                    for h in dict_asl[letra]:
                        d = np.mean(
                            [distance.euclidean(a, b) for a, b in zip(hpred, h)]
                        )
                        if d < dmin:
                            dmin = d
                            letramin = letra
                preds.append(letramin)
                print(f"letramin = {letramin}, preds = {preds}")

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
        if len(self.frames) > 0:
            if _bytes:
                img = cv2.imencode(".png", self.frames[-1])[1].tobytes()
            else:
                img = self.frames[-1]
        else:
            with open("images/not_found.jpeg", "rb") as f:
                img = f.read()
        return img
