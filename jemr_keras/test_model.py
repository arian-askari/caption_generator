# coding=utf-8
"""
This code deals with testing the created model
"""
from __future__ import print_function

import cPickle
import os

import nltk
import numpy as np
from keras.preprocessing import sequence

from jemr_model import JemrModel

os.environ['CUDA_VISIBLE_DEVICES'] = ''

cg = JemrModel()


def process_caption_text(caption):
    """
    process caption by removing begin and end symbols from beginning and end of caption
    :param caption: caption text
    :return: processed caption
    """
    caption_split = caption.split()
    processed_caption = caption_split[1:]
    try:
        end_index = processed_caption.index('<end>')
        processed_caption = processed_caption[:end_index]
    except ValueError:
        pass
    return " ".join([word for word in processed_caption])


def get_best_caption_text(captions):
    """
    find the text of candidate caption with the highest probability of being correct caption
    :param captions: all caption candidates with their probabilities
    :return: text of best caption candidate
    """
    captions.sort(key=lambda l: l[1])
    best_caption = captions[-1][0]
    return " ".join([cg.index_word[index] for index in best_caption])


def get_all_captions_texts(captions):
    """
    find the text of all candidate caption regardless of their highest probability of being
    correct caption
    :param captions: all caption candidates with their probabilities
    :return: text of all caption candidates
    """
    final_captions = []
    captions.sort(key=lambda l: l[1])
    for caption in captions:
        text_caption = " ".join([cg.index_word[index] for index in caption[0]])
        final_captions.append([text_caption, caption[1]])
    return final_captions


def generate_captions_probs(model, image, beam_size):
    """
    run the model and predict the caption candidates with the highest probabilities of being correct caption
    :param model: the neural network model
    :param image: the given image to be captioned
    :param beam_size: size of beam
    :return: captions with their probabilities
    """
    start = [cg.word_index['<start>']]
    captions = [[start, 0.0]]
    while len(captions[0][0]) < cg.max_cap_len:
        temp_captions = []
        for caption in captions:
            partial_caption = sequence.pad_sequences([caption[0]], maxlen=cg.max_cap_len, padding='post')
            next_words_pred = model.predict([np.asarray([image]), np.asarray(partial_caption)])[0]
            next_words = np.argsort(next_words_pred)[-beam_size:]
            for word in next_words:
                new_partial_caption, new_partial_caption_prob = caption[0][:], caption[1]
                new_partial_caption.append(word)
                new_partial_caption_prob += next_words_pred[word]
                temp_captions.append([new_partial_caption, new_partial_caption_prob])
        captions = temp_captions
        captions.sort(key=lambda l: l[1])
        captions = captions[-beam_size:]

    return captions


def test_model(weight, img_name, beam_size=3):
    encoded_images = cPickle.load(open("encoded_images.p", "rb"))
    model = cg.create_model(ret_model=True)
    model.load_weights(weight)

    image = encoded_images[img_name]
    captions = generate_captions_probs(model, image, beam_size)
    return process_caption_text(get_best_caption_text(captions))


# return [process_caption(caption[0]) for caption in get_all_captions(captions)]

def bleu_score(hypotheses, references):
    """
    compute blue score
    :param hypotheses: generated texts
    :param references: correct texts
    :return: blue score of the method
    """
    return nltk.translate.bleu_score.corpus_bleu(references, hypotheses)


def test_model_on_images(weight, img_dir, beam_size=3):
    captions = {}
    with open(img_dir, 'rb') as f_images:
        imgs = f_images.read().strip().split('\n')
    encoded_images = cPickle.load(open("encoded_images.p", "rb"))
    model = cg.create_model(ret_model=True)
    model.load_weights(weight)

    f_pred_caption = open('predicted_captions.txt', 'wb')

    for count, img_name in enumerate(imgs):
        print("Predicting for image: " + str(count))
        image = encoded_images[img_name]
        image_captions_indices = generate_captions_probs(model, image, beam_size)
        print("all captions",
              [process_caption_text(caption[0]) for caption in get_all_captions_texts(image_captions_indices)])
        print("best caption", process_caption_text(get_best_caption_text(image_captions_indices)))
        best_caption = process_caption_text(get_best_caption_text(image_captions_indices))
        captions[img_name] = best_caption
        print(img_name + " : " + str(best_caption))
        f_pred_caption.write(img_name + "\t" + str(best_caption) + "\n")
        f_pred_caption.flush()
    f_pred_caption.close()

    f_captions = open('Flickr8k_text/Flickr8k.token.txt', 'rb')
    captions_text = f_captions.read().strip().split('\n')
    image_captions_pair = {}
    for row in captions_text:
        row = row.split("\t")
        row[0] = row[0][:len(row[0]) - 2]
        try:
            image_captions_pair[row[0]].append(row[1])
        except IndexError:
            image_captions_pair[row[0]] = [row[1]]
    f_captions.close()

    hypotheses = []
    references = []
    for img_name in imgs:
        hypothesis = captions[img_name]
        reference = image_captions_pair[img_name]
        hypotheses.append(hypothesis)
        references.append(reference)

    return bleu_score(hypotheses, references)


if __name__ == '__main__':
    weight_dir = 'weights-improvement-03.hdf5'
    test_image = '3155451946_c0862c70cb.jpg'
    test_img_dir = 'Flickr8k_text/Flickr_8k.testImages.txt'
    # print test_model(weight, test_image)
    print(test_model_on_images(weight_dir, test_img_dir))
