# coding=utf-8
"""
Encoding model for the visual modalities
"""
from keras import Sequential
from keras.applications import VGG16
from keras.layers import Dense, RepeatVector, LSTM, TimeDistributed, Activation, Merge, Flatten

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 12 / 20 / 17


class ImageEncoder(object):
    """
    class for creating encoder for visual modality
    """

    def __init__(self, vocab_size, max_cap_len, embedding_dim, num_gen_captions):
        self.vocab_size = vocab_size
        self.max_cap_len = max_cap_len
        self.embedding_dim = embedding_dim
        self.num_gen_captions = num_gen_captions

    def create_layer(self, embedding_layer, pre_run_cnn=True):
        """
        create layer of encoder for visual component
        :param embedding_layer: embedding layer with weights shared in all of network
        :param pre_run_cnn: if False, CNN will during running the session, otherwise, it gets image code generated
        separately
        :return: layer of encoder for visual component
        """
        image_model = Sequential()
        if not pre_run_cnn:
            base_model = VGG16(include_top=False, input_shape=(224, 224, 3))
            base_model.trainable = False
            image_model.add(base_model)
            image_model.add(Flatten())
        image_model.add(Dense(self.embedding_dim, input_dim=4096, activation='relu'))
        image_model.add(RepeatVector(self.max_cap_len))

        lang_model = Sequential()
        lang_model.add(embedding_layer)
        lang_model.add(LSTM(256, return_sequences=True))
        lang_model.add(TimeDistributed(Dense(self.embedding_dim)))

        # layers recursively called with the same weight:
        lstm_ = LSTM(1000)
        dense_ = Dense(self.vocab_size)
        dense_emb_ = Dense(self.embedding_dim, activation='relu')
        act_ = Activation('softmax')
        rep_ = RepeatVector(self.max_cap_len)

        # recursively call LSTM, FC layers to generate 'num_gen_captions' number of captions
        model_lstm = [lang_model]
        for i in range(self.num_gen_captions):
            if len(model_lstm) == 1:
                aggregated_model = model_lstm[0]
            else:
                aggregated_model = Merge(model_lstm)
            model1 = Sequential(
                [Merge([image_model, aggregated_model], mode='concat'),
                 lstm_,
                 dense_,
                 act_,
                 dense_emb_,
                 rep_]
            )
            model_lstm.append(model1)
        return Merge(model_lstm)
