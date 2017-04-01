import cv2
import traceback

from core.plate_locate import PlateLocate
from util.figs import imshow

def test_plate_locate():
    print("Testing Plate Locate")
    #try:
    file = "resources/image/test.jpg"

    src = cv2.imread(file)

    result = []
    plate = PlateLocate()
    plate.setDebug(False)
    plate.setLifemode(True)

    if plate.plateLocate(src, result) == 0:
        for res in result:
            imshow("plate locate", res)

    #except:
     #   traceback.print_exc()


def test_plate_judge():
    print("Testing Plate Judge")

def test_plate_detect():
    pass

def test_char_segment():
    pass

def test_chars_identify():
    pass

def test_chars_recognise():
    pass

def test_plate_recogize():
    pass
