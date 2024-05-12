# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import nltk
from nltk.corpus import gutenberg

alice = gutenberg.raw(fileids='carroll-alice.txt')
default_st = nltk.sent_tokenize
alice_sentences = default_st(text=alice)
default_wt = nltk.word_tokenize
alice_words = [default_wt(sentence.lower()) for sentence in alice_sentences]
print(len(alice_sentences))
len_alice = len(alice_sentences)

hamlet = gutenberg.raw(fileids='shakespeare-hamlet.txt')
hamlet_sentences = default_st(text=hamlet)
hamlet_words = [default_wt(sentence.lower()) for sentence in hamlet_sentences]
print(len(hamlet_sentences))
len_hamlet = len(hamlet_sentences)

sentences = alice_words + hamlet_words
len_data = len(sentences)

word2index_map = {}
index = 0
for sent in sentences:
    for word in sent:
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1

index2word_map = {index: word for word, index in word2index_map.items()}

vocabulary_size = len(index2word_map)
# Generate skip-gram pairs
skip_gram_pairs = []
win_size = 3
for sent in sentences:
    for i in range(win_size, len(sent)-win_size):
        for wn in range(win_size):
            word_context_pair = [[word2index_map[sent[i-wn]],
                                word2index_map[sent[i+wn]]],
                                word2index_map[sent[i]]]
            skip_gram_pairs.append([word_context_pair[1],
                                  word_context_pair[0][0]])
            skip_gram_pairs.append([word_context_pair[1],
                                 word_context_pair[0][1]])

batch_size = 64
embedding_dimension = 3 # Three was chosen for visualization.
negative_samples = 8
LOG_DIR = "/home/simon/Firma/AI/HegelMachine/HegelPython/logs/word2vec_intro/" # Use absolute path.

def get_skipgram_batch(batch_size):
    instance_indices = list(range(len(skip_gram_pairs)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [skip_gram_pairs[i][0] for i in batch]
    y = [[skip_gram_pairs[i][1]] for i in batch]
    return x, y

# Input data, labels
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

# Embedding lookup table currently only implemented in CPU
with tf.name_scope("embeddings"):
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_dimension],
                          -1.0, 1.0), name='embedding')
    # This is essentialy a lookup table
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Create variables for the NCE loss
nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_dimension],
                            stddev=1.0 / math.sqrt(embedding_dimension)))
nce_biases = tf.Variable(tf.