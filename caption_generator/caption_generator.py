import cPickle as pickle

import numpy as np
import pandas as pd
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation
from keras.models import Sequential
from keras.preprocessing import image, sequence
from keras.utils import plot_model

EMBEDDING_DIM = 128


class CaptionGenerator:
    def __init__(self):
        self.max_cap_len = None
        self.vocab_size = None
        self.index_word = None
        self.word_index = None
        self.total_samples = None
        self.encoded_images = pickle.load(open("encoded_images.p", "rb"))
        self.variable_initializer()

    def variable_initializer(self):
        """
        initializes Total number of samples (self.total_samples),
                    Vocabulary size (self.vocab_size),
                    Maximum caption length (self.max_cap_len),
                    word to index dictionary (self.word_index), and
                    index to word dictionary (self.index_word)
        """
        df = pd.read_csv('Flickr8k_text/flickr_8k_train_dataset.txt', delimiter='\t')
        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        for i in range(nb_samples):
            x = iter.next()
            caps.append(x[1][1])

        self.total_samples = 0
        for text in caps:
            self.total_samples += len(text.split()) - 1
        print "Total samples : " + str(self.total_samples)

        words = [txt.split() for txt in caps]
        unique = []
        for word in words:
            unique.extend(word)

        unique = list(set(unique))
        self.vocab_size = len(unique)
        self.word_index = {}
        self.index_word = {}
        for i, word in enumerate(unique):
            self.word_index[word] = i
            self.index_word[i] = word

        max_len = 0
        for caption in caps:
            if len(caption.split()) > max_len:
                max_len = len(caption.split())
        self.max_cap_len = max_len
        print "Vocabulary size: " + str(self.vocab_size)
        print "Maximum caption length: " + str(self.max_cap_len)
        print "Variables initialization done!"

    def data_generator(self, batch_size=32):
        """
        Given training data file, yields batches of training data
        :param batch_size: batch size
        :return: batches of training data, which is an array of images, partial captions and word
         next to partial captions, all encoded.
        """
        partial_caps = []
        next_words = []
        images = []
        print "Generating data..."
        gen_count = 0
        df = pd.read_csv('Flickr8k_text/flickr_8k_train_dataset.txt', delimiter='\t')
        nb_samples = df.shape[0]
        it_dataset = df.iterrows()
        caps = []
        imgs = []
        for i in range(nb_samples):
            x = it_dataset.next()
            caps.append(x[1][1])
            imgs.append(x[1][0])

        total_count = 0
        while 1:

            # initialize image_counter at the beginning of each epoch
            image_counter = -1

            # for each instance in the training data:
            for text in caps:

                # handle images:
                image_counter += 1
                current_image = self.encoded_images[imgs[image_counter]]

                # handle captions:
                caption_words = text.split()
                for i in range(len(caption_words) - 1):
                    total_count += 1

                    # create partial caption from 0-i'th caption words
                    partial = [self.word_index[txt] for txt in caption_words[:(i + 1)]]

                    # store partial captions
                    partial_caps.append(partial)

                    # create one-hot vector for the next caption word:
                    next_w = np.zeros(self.vocab_size)
                    next_w[self.word_index[caption_words[i + 1]]] = 1

                    # store the one-vector of the next word
                    next_words.append(next_w)

                    # store the current_image once for each word
                    images.append(current_image)

                    # if enough number of caption words to make a batch
                    if total_count >= batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=self.max_cap_len, padding='post')
                        gen_count += 1
                        if gen_count % 1000 == 0:
                            print "yielding count: " + str(gen_count)
                        yield [[images, partial_caps], next_words]

                        # initialize for the next batch:
                        total_count = 0
                        partial_caps = []
                        next_words = []
                        images = []

    @staticmethod
    def load_image(path):
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        return np.asarray(x)

    def create_model(self, ret_model=False):
        # base_model = VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
        # base_model.trainable=False
        image_model = Sequential()
        # image_model.add(base_model)
        # image_model.add(Flatten())
        image_model.add(Dense(EMBEDDING_DIM, input_dim=4096, activation='relu'))
        image_model.add(RepeatVector(self.max_cap_len))

        lang_model = Sequential()
        lang_model.add(Embedding(self.vocab_size, 256, input_length=self.max_cap_len))
        lang_model.add(LSTM(256, return_sequences=True))
        lang_model.add(TimeDistributed(Dense(EMBEDDING_DIM)))

        l_ = LSTM(1000, return_sequences=False)
        d_ = Dense(self.vocab_size)
        de_ = Dense(EMBEDDING_DIM, activation='relu')
        a_ = Activation('softmax')
        r_ = RepeatVector(self.max_cap_len)

        model_lstm = [lang_model]
        for i in range(3):
            model1 = Sequential()
            if len(model_lstm) == 1:
                aggregated_model = model_lstm[0]
            else:
                aggregated_model = Merge(model_lstm, mode='sum')
            model1.add(Merge([image_model, aggregated_model], mode='concat'))
            model1.add(l_)
            model1.add(d_)
            model1.add(a_)
            model1.add(de_)
            model1.add(r_)
            model_lstm += [model1]

        model_desc = Sequential()
        model_desc.add(Merge([image_model, Merge(model_lstm, mode='sum')], mode='concat'))
        model_desc.add(l_)
        model_desc.add(d_)
        model_desc.add(a_)
        # model3.add(Dense(EMBEDDING_DIM, activation='relu'))

        model = model_desc
        plot_model(model, to_file='exampleRec.png')

        print "Model created!"

        if ret_model:
            return model

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def get_word(self, index):
        return self.index_word[index]
