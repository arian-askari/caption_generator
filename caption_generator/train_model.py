import os

from keras import callbacks
from keras.callbacks import ModelCheckpoint

import caption_generator

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def train_model(weight=None, batch_size=32, epochs=10):
    cg = caption_generator.CaptionGenerator()
    model = cg.create_model()

    if weight is not None:
        model.load_weights(weight)

    file_name = 'weights-improvement-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
    tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    data_iterator = cg.data_generator(batch_size=batch_size)

    # # Function to display the target and prediciton
    # def testmodel(epoch, logs):
    #     predx, predy = next(data_iterator)
    #     predout = model.predict(
    #         predx,
    #         batch_size=batch_size
    #     )
    #     print("Input\n")
    #     print(predx)
    #     print("Target\n")
    #     print(predy)
    #     print("Prediction\n")
    #     print(predout)
    #
    # # Callback to display the target and prediciton
    # testmodelcb = callbacks.LambdaCallback(on_epoch_end=testmodel)

    callbacks_list = [checkpoint, tbCallBack]
    model.fit_generator(data_iterator, steps_per_epoch=cg.total_samples / batch_size,
                        epochs=epochs, verbose=1, callbacks=callbacks_list)
    try:
        model.save('Models/WholeModel.h5', overwrite=True)
        model.save_weights('Models/Weights.h5', overwrite=True)
    except:
        print "Error in saving model."
    print "Training complete...\n"


if __name__ == '__main__':
    train_model(epochs=50)
