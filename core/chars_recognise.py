from .chars_segment import CharsSegment
from .chars_identify import CharsIdentify

class CharsRecognise(object):
    def __init__(self):
        self.charsSegment = CharsSegment()

    def charsRecognise(self, plate, plate_license):
        chars = []
        result = self.charsSegment.charsSegment(plate, chars)
        if result == 0:
            for c in chars:
                plate_license += CharsIdentify.identify(c)

        if len(plate_license) < 7:
            return -1

        return result