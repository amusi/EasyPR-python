import cv2
import numpy as np

from .base import Plate
from .plate_locate import PlateLocate
from .plate_judge import PlateJudge

from .core_func import *
from util.figs import imwrite, imshow

class PlateDetect(object):
    def __init__(self):
        self.m_plateLocate = PlateLocate()
        self.m_maxPlates = 3

    def setPDLifemode(self, param):
        self.m_plateLocate.setLifemode(param)

    def plateDetect(self, src, res, showDetectArea=True, index=0):
        color_plates = []
        sobel_plates = []
        color_result_plates = []
        sobel_result_plates = []

        color_find_max = self.m_maxPlates

        self.m_plateLocate.plateColorLocate(src, color_plates, index)

        color_result_plates = PlateJudge().judge(color_plates)

        for plate in color_result_plates:
            plate.plate_type = "COLOR"
            res.append(plate)

        self.m_plateLocate.plateSobelLocate(src, sobel_plates, index)

        sobel_result_plates = PlateJudge().judge(sobel_plates)

        for plate in color_result_plates:
            plate.bColored = False
            plate.plate_type = "SOBEL"
            res.append(plate)

        return 0




