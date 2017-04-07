import cv2
import numpy as np

from .plate_locate import PlateLocate

from .core_func import *
from util.figs import imwrite, imshow

class PlateDetect(object):
    def __init__(self):
        self.m_plateLocate = PlateLocate()
        self.m_maxPlates = 3

    def setPDLifemode(self, param):
        self.m_plateLocate.setLifemode(param)

    def plateDetect(self, src, result, showDetectArea=True, index=0):
        pass