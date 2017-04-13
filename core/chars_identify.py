from .base import Singleton

from train.cnn_train import eval_model
from train.net.Lenet import Lenet

class CharsIdentify(Singleton):

    def __init__(self):

        self.model = Lenet()
        self.model.compile()

    def identify(self, images):

        pred = eval_model(self.model.pred_labels,
                               {self.model.x: images, self.model.keep_prob: 1},
                               model_dir="train/model/chars/models/")

        return pred