import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import gutenberg

default_st = nltk.sent_tokenize
default_wt = nltk.word_tokenize

alice = gutenberg.raw(fileids='carroll-alice.txt')
alice_sentences = default_st(text=alice)
alice_tuples = [nltk.pos_tag(default_wt(sentence)) for sentence in alice_sentences]
alice_words = [[word[0] for word in sentence] for sentence in alice_tuples]
alice_tags = [[word[1] for word in sentence] for sentence in alice_tuples]
alice_len = len(alice_sentences)

melville = gutenberg.raw(fileids='melville-moby_dick.txt')
melville_sentences = default_st(melville)
mel