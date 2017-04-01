from plate import *

main_menu = (
    '-'*8 + '\n' +
    '选择测试:\n' +
    '1. 功能测试;\n' +
    '2. 准确度测试;(Invalid)\n' +
    '-'*8 + '\n'
)

test_menu = (
    '-'*8 + '\n' +
    '功能测试:\n' +
    '1. test plate_locate(车牌定位);\n' +
    '2. test plate_judge(车牌判断);\n' +
    '3. test plate_detect(车牌检测);\n' +
    '4. test chars_segment(字符分隔);\n' +
    '5. test chars_identify(字符鉴别);\n' +
    '6. test chars_recognise(字符识别);\n' +
    '7. test plate_recognize(车牌识别);\n' +
    '8. test all(测试全部);\n' +
    '9. 返回;\n' +
    '-'*8 + '\n'
)

accuary_menu = (
    '-'*8 + '\n' +
    '准确度测试:\n' +
    '1. scm测试;\n' +
    '2. ann测试;\n' +
    '-'*8 + '\n'
)

def command_line_handler():
    #try:
    while(1):
        print(main_menu)
        select = input()
        main_op[select]()

    #except Exception as e:
    #   print("Exit")

def test_main():
    #try:
    while(1):
        print(test_menu)
        select = input()
        test_op[select]()

    #except Exception as e:
    #    print("Exit")


def test_accuary():
    print(accuary_menu)
    pass

main_op = {
    '1': test_main,
    '2': test_accuary,
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
