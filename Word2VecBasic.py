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
       