import cv2
import numpy as np

class PlateLocate(object):
    def __init__(self):
        self.m_GaussianBlurSize = 5
        self.m_MorphSizeWidth = 17
        self.m_MorphSizeHeight = 3

        self.m_error = 0.9
        self.m_aspect = 3.75
        self.m_verifyMin = 1
        self.m_verifyMax = 24

        self.m_angle = 60

        self.m_debug = 1

    def setLifemode(self, param):
        pass

    def setDebug(self):
        pass