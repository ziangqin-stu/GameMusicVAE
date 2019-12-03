"""
Base models schema
"""


class BaseEncoder(object):
    """
    Abstract encoder class
    """
    def output_depth(self):
        pass

    def build(self, hparams, is_training=True):
        pass

    def encode(self, sequence, sequence_length):
        pass
