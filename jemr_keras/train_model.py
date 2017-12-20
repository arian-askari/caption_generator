# coding=utf-8
"""
    This code trains model for the created model
"""
from __future__ import print_function

import os

from keras import callbacks
from keras.callbacks import ModelCheckpoint

from jemr_model import JemrModel

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def train_model(weight=None, batch_size=32, epochs=10):
    """
    train the model
    :param weight: weights of the network
    :param batch_size: batch size for training network
    :param epochs: number of epochs
    """
    cg = JemrModel()
    model = cg.create_model()

    if weight is not None:
        model.load_weights(weight)

    file_name = 'weights-improvement-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
    tb_call_back = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1, write_images=True)

    data_iterator = cg.data_generator(batch_size=batch_size)

    callbacks_list = [checkpoint, tb_call_back]
    model.fit_generator(data_iterator, steps_per_epoch=cg.total_samples / batch_size,
                        epochs=epochs, verbose=1, callbacks=callbacks_list)
    try:
        model.save('Models/WholeModel.h5', overwrite=True)
        model.save_weights('Models/Weights.h5', overwrite=True)
    except ImportError:
        print("Error in saving model.")
    print("Training complete...\n")


if __name__ == '__main__':
    train_model(epochs=50)
