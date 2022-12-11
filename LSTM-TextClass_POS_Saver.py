import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import gutenberg
import os
import glob
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.model_selection import train_test_split


def data_builder(file_id):
    d = gutenberg.raw(fileids=file_id)
    d_sentences = default_st(text=d)
    d_tuples = [nltk.pos_tag(default_wt(sentence)) for sentence in d_sentences]
    d_words = [[word[0] for word in sentence] for sentence in d_tuples]
    d_tags = [[word[1] for word in sentence] for sentence in d_tuples]
    d_len = len(d_sentences)
    return d_sentences, d_words, d_tags, d_len


def get_sentence_batch(batch_size, data_x,
                       data_y, data_seqlens, data_x_tags):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [[word2index_map[word] for word in data_x[i]]
         for i in batch]
    x2 = [[word2index_map_tags[word] for word in data_x_tags[i]]
         for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    return x, y, seqlens, x2


SAVE_PATH = 'logs/TextClass/'

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

default_st = nltk.sent_tokenize
default_wt = nltk.word_tokenize
batch_size = 100
embedding_dimension = 64
embedding_dimension_tags = 32
num_classes = 3
hidden_layer_size = 128
num_LSTM_layers = 4
hidden_layer_size_tags = 64
num_LSTM_layers_tags = 2
epochs = 201
test_ratio = 0.2

alice_sentences, alice_words, alice_tags, alice_len = data_builder('carroll-alice.txt')
mel