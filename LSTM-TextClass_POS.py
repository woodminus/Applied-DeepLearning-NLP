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
melville_tuples = [nltk.pos_tag(default_wt(sentence)) for sentence in melville_sentences]
melville_words = [[word[0] for word in sentence] for sentence in melville_tuples]
melville_tags = [[word[1] for word in sentence] for sentence in melville_tuples]
melville_len = len(melville_sentences)

austin = gutenberg.raw(fileids='austen-sense.txt')
austin_sentences = default_st(austin)
austin_tuples = [nltk.pos_tag(default_wt(sentence)) for sentence in austin_sentences]
austin_words = [[word[0] for word in sentence] for sentence in austin_tuples]
austin_tags = [[word[1] for word in sentence] for sentence in austin_tuples]
austin_len = len(austin_sentences)

data = np.array(alice_words + melville_words + austin_words)
data_tags = np.array(alice_tags + melville_tags + austin_tags)
data_len = len(data)

max_len = 0
for sent in data:
    if len(sent)>max_len:
        max_len = len(sent)

batch_size = 100
embedding_dimension = 64
embedding_dimension_tags = 32
num_classes = 3
hidden_layer_size = 128
num_LSTM_layers = 4
hidden_layer_size_tags = 64
num_LSTM_layers_tags = 2
#element_size = 1
epochs = 1500

seqlens = []

for sentence_id in range(data_len):
    seqlens.append(len(data[sentence_id]))

    if len(data[sentence_id]) < max_len:
        pads = ['PAD']*(max_len-len(data[sentence_id]))
   