import cv2

from core.plate_locate import PlateLocate

def test_plate_locate():
    print("Testing Plate Locate")
    file = "resources/image/test.jpg"

    src = cv2.imread(file)
    result = []
    plate = PlateLocate()
    plate.setDebug(1)
    plate.setLifemode(True)

    if plate.plateLocate(src, result) == 0:
        for res in result:
            cv2.imshow("plate locate", res)
            cv2.waitKey(0)
        cv2.destroyWindow("plate locate")


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
