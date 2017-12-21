# coding=utf-8
"""

"""
from keras import Sequential
from keras.layers import Merge, Bidirectional, Dropout, LSTM

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 12 / 21 / 17


class Matching(object):
    """
    class to create matching layer
    """

    def __init__(self, embedding_dim, max_cap_len, match_embed_size):
        self.embedding_dim = embedding_dim
        self.max_cap_len = max_cap_len
        self.match_embed_size = match_embed_size

    def create_layer(self, left_layer, right_layer):
        """
        create matching layer
        :param left_layer: one side of matching layer that will contain one of the embedding vector
        :param right_layer: other side of matching layer that will contain one of the embedding vector
        :return:
        """
        # bidirectional layer with weights shared over siamese network
        bidirectional_layer = Bidirectional(
            LSTM(self.match_embed_size, input_shape=(self.max_cap_len, self.embedding_dim)),
            merge_mode="sum")

        model_left = Sequential()
        model_left.add(left_layer)
        model_left.add(bidirectional_layer)
        model_left.add(Dropout(0.3))

        model_right = Sequential()
        model_right.add(right_layer)
        model_right.add(bidirectional_layer)
        model_right.add(Dropout(0.3))

        return Merge([model_left, model_right])
