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
        data[sentence_id] = data[sentence_id] + pads

### tags ###
for sentence_id in range(data_len):
    if len(data_tags[sentence_id]) < max_len:
        pads = ['PAD']*(max_len-len(data_tags[sentence_id]))
        data_tags[sentence_id] = data_tags[sentence_id] + pads

####

labels = [2] * alice_len + [1] * melville_len + [0] * austin_len
# labels = [2] * 100 + [1] * 100 + [0] * 100
# labels_hot = tf.one_hot(labels, depth=num_classes)

for i in range(len(labels)):
    label = labels[i]
    one_hot_encoding = [0]*num_classes
    one_hot_encoding[label] = 1
    labels[i] = one_hot_encoding

word2index_map = {}
index = 0
for sent in data:
    for word in sent:
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1

index2word_map = {index: word for word, index in word2index_map.items()}

vocabulary_size = len(index2word_map)

#### tags ####

word2index_map_tags = {}
index = 0
for sent in data_tags:
    for tag in sent:
        if tag not in word2index_map_tags:
            word2index_map_tags[tag] = index
            index += 1

index2word_map_tags = {index: tag for tag, index in word2index_map_tags.items()}

vocabulary_size_tags = len(index2word_map_tags)

###

train_size = int(data_len/2) # has to be integer for slicing array
data_indices = list(range(len(data)))
np.random.shuffle(data_indices)
d