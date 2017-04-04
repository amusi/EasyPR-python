import cv2
import traceback

from core.plate_locate import PlateLocate
from core.plate_detect import PlateDetect
from core.chars_segment import CharsSegment

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
    # try:
    file = "resources/image/test.jpg"

    src = cv2.imread(file)

    result = []
    plate = PlateLocate()
    plate.setDebug(False)
    plate.setLifemode(True)

    if plate.plateLocate(src, result) == 0:
        for res in result:
            imshow("plate locate", res)

    # except:
    #   traceback.print_exc()

def test_plate_detect():
    print("Testing Plate Detect")

    file = "resources/image/plate_detect.jpg"

    src = cv2.imread(file)

    result = []
    plate = CharsSegment()


    if plate.charsSegment(src, result) == 0:
        for i in range(len(result)):
            imshow("plate locate " + str(i), result[i])


def test_char_segment():
    print("Testing Chars Segment")

    file = "resources/image/chars_segment.jpg"

    src = cv2.imread(file)
    imshow("src", src)
    result = []
    plate = CharsSegment()

    if plate.charsSegment(src, result) == 0:
        for i in range(len(result)):
            imshow("plate segment " + str(i), result[i])

def test_chars_identify():
    print("Testing Chars Segment")

    file = "resources/image/chars_identify.jpg"

    src = cv2.imread(file)
    imshow("src", src)
    result = []
    license = []
    plate = CharsSegment()

    if plate.charsSegment(src, result) == 0:
        for i in range(len(result)):
            license.append()

def test_chars_recognise():
    pass

def test_plate_recogize():
    pass
