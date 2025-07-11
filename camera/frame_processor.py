import cv2
import numpy as np

class FrameProcessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    def enhance_night_vision(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def resize_frame(self, frame, target_size):
        return cv2.resize(frame, target_size)
