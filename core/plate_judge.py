from .base import Singleton

from train.cnn_train import eval_model
from train.net.judgenet import Judgenet

class PlateJudge(Singleton):

    def __init__(self):

        self.model = Judgenet()
        self.model.compile()

    def judge(self, images):

        pred = eval_model(self.model.pred_labels,
                               {self.model.x: images, self.model.keep_prob: 1},
                               model_dir="train/model/whether_car/models/")

        return pred