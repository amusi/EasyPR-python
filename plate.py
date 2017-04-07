import cv2
import numpy as np

from core.plate_locate import PlateLocate
from core.plate_detect import PlateDetect
from core.chars_segment import CharsSegment
from core.chars_recognise import CharsRecognise
from core.plate_judge import PlateJudge

from train.cnn_train import Lenet
from train.dataset import DataSet

from util.figs import imshow
from util.read_etc import index2str

def test_plate_locate():
    print("Testing Plate Locate")

    file = "resources/image/test.jpg"

    src = cv2.imread(file)

    result = []
    plate = PlateLocate()
    plate.setDebug(False)
    plate.setLifemode(True)

    if plate.plateLocate(src, result) == 0:
        for res in result:
            imshow("plate locate", res)


def test_plate_judge():
    print("Testing Plate Judge")

    file = "resources/image/test.jpg"

    src = cv2.imread(file)

    result = []
    plate = PlateLocate()
    plate.setDebug(False)
    plate.setLifemode(True)

    if plate.plateLocate(src, result) == 0:
        for res in result:
            imshow("plate judge", res)

    judge_result = []
    PlateJudge.plateJudge(result, judge_result)


def test_plate_detect():
    print("Testing Plate Detect")

    file = "resources/image/plate_detect.jpg"

    src = cv2.imread(file)

    result = []
    pd = PlateDetect()
    pd.setPDLifemode(True)

    if pd.plateDetect(src, result):
        for res in result:
            imshow("plate detect", res.plate_image)



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
    plate_license = []
    plate = CharsSegment()

    if plate.charsSegment(src, result) == 0:
        for i in range(len(result)):
            license.append()

    print("Plate License: ", "苏E771H6")
    print("Plate Identify: ", plate_license)
    if (plate_license == "苏E771H6"):
        print("Identify Correct")
    else:
        print("Identify Not Correct")

def test_chars_recognise():
    print("Testing Chars Segment")

    file = "resources/image/chars_recognise.jpg"

    src = cv2.imread(file)
    imshow("src", src)

    plate_license = ""

    cr = CharsRecognise()

    if cr.charsRecognise(src, plate_license):
        print("Chars Recognise: ", plate_license)


def test_plate_recogize():
    pass

def test_cnn(image_path=None, label=None):
    if image_path == None:
        image_path = input("Please input image path: ")
        if image_path == "":
            image_path = 'resources/train_data/chars/0/4-3.jpg'

    print("Image Path: ", image_path)
    src = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)[..., None]
    imshow("src", src)
    batch_size = 1
    params = {
        'image_size': 20,
        'batch_size': batch_size,
        'prob': 1,
        'lr': 0.01,
        'max_steps': 1,
        'log_dir': 'train/model/chars/'
    }

    model = Lenet(params)
    model.compile()
    pred, _ = model.predict(src)
    if label == None:
        print("Pred: ", pred[0])
    else:
        print("Label: {}, Pred: {}".format(label, pred[0]))


def test_cnn_val():
    batch_size = 1

    dataset_params = {
        'batch_size': batch_size,
        'path': 'resources/train_data/chars',
        'labels_path': 'resources/train_data/chars_list_val.pickle',
        'thread_num': 3
    }
    val_dataset_reader = DataSet(dataset_params)


    params = {
        'image_size': 20,
        'batch_size': batch_size,
        'prob': 1,
        'lr': 0.01,
        'max_steps': 1,
        'log_dir': 'train/model/chars/'
    }
    model = Lenet(params)
    model.compile()
    total_time = []
    total_acc = []

    for i in range(val_dataset_reader.record_number):
        image, label = val_dataset_reader.batch()

        pred, time = model.predict(image)

        total_acc.append(pred[0] == label[0])
        total_time.append(time)
        print("Label: {}({}), Pred: {}({})".format(label[0], index2str[label[0]], pred[0], index2str[pred[0]]))
        #imshow("tmp", image[0])

    print("Mean time: {} sec".format(np.mean(total_time)))
    print("Accuary: {:.2f}%".format(np.sum(total_acc) / val_dataset_reader.record_number * 100))

