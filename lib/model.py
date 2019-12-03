"""

"""

from lib.base_model import *
from rnn_utils import *


class LSTMEncoder(BaseEncoder):

    def output_depth(self):
        return self._cell.output_size

    def build(self, hparams, is_training=True):
        self.is_training = is_training
        self.lstm = rnn_utils.lstm_layer()