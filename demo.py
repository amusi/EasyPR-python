from plate import *
from train.cnn_train import main

main_menu = (
    '-'*8 + '\n' +
    '选择测试:\n' +
    '1. 功能测试;\n' +
    '2. 准确度测试;(Invalid)\n' +
    '3. SVM训练;(Invalid)\n' +
    '4. CNN相关;\n' +
    '-'*8
)

test_menu = (
    '-'*8 + '\n' +
    '功能测试:\n' +
    '1. test plate_locate(车牌定位);\n' +
    '2. test plate_judge(车牌判断);(Invalid)\n' +
    '3. test plate_detect(车牌检测);(Invalid)\n' +
    '4. test chars_segment(字符分隔);\n' +
    '5. test chars_identify(字符鉴别);\n' +
    '6. test chars_recognise(字符识别);\n' +
    '7. test plate_recognize(车牌识别);(Invalid)\n' +
    '-'*8
)

accuary_menu = (
    '-'*8 + '\n' +
    '批量测试:\n' +
    '1. 普通情况测试;\n' +
    '2. 极端情况测试;\n' +
    '-'*8
)

cnn_menu = (
    '-'*8 + '\n' +
    'cnn相关:\n' +
    '1. cnn训练;\n' +
    '2. cnn测试(测试集);\n' +
    '-'*8
)

def command_line_handler():
    while(1):
        print(main_menu)
        select = input()
        main_op[select]()


def test_main():
    while(1):
        print(test_menu)
        select = input()
        test_op[select]()


def test_accuary():
    print(accuary_menu)
    pass


def svm_train():
    pass


def cnn_rel():
    print(cnn_menu)
    select = input()
    if select == '1':
        main()
    elif select == '2':
        test_cnn_val()
    else:
        print("Error choice")

main_op = {
    '1': test_main,
    '2': test_accuary,
    '3': svm_train,
    '4': cnn_rel
}

test_op = {
    '1': test_plate_locate,
    '2': test_plate_judge,
    '3': test_plate_detect,
    '4': test_char_segment,
    '5': test_chars_identify,
    '6': test_chars_recognise,
    '7': test_plate_recogize,
}

if __name__ == "__main__":
    command_line_handler()
