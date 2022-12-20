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
melville_sentences, melville_words, melville_tags, melville_len = data_builder('melville-moby_dick.txt')
austin_sentences, austin_words, austin_tags, austin_len = data_builder('austen-sense.txt')

data = np.array(alice_words + melville_words + austin_words)
data_tags = np.array(alice_tags + melville_tags + austin_tags)
data_len = len(data)

max_len = 0
for sent in data:
    if len(sent)>max_len:
        max_len = len(sent)

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

train_x, test_x, train_y, test_y, train_seqlens, test_seqlens, train_x_tags, test_x_tags, = \
    train_test_split(data, labels, seqlens, data_tags, test_size=test_ratio, random_state=42)


#tensorflow
_inputs = tf.placeholder(tf.int32, shape=[batch_size, max_len], name='Input')
_labels = tf.placeholder(tf.float32, shape=[batch_size, num_classes], name='Labels')
_seqlens = tf.placeholder(tf.int32, shape=[batch_size], name='Seqlens')
_inputs_tags = tf.placeholder(tf.int32, shape=[batch_size, max_len], name='Input_tags')
_miss = tf.placeholder(tf.string, shape=[None], name="MissClass")

global_step = tf.Variable(0, trainable=False, name='global_step')
increment_global_step = tf.assign_add(global_step, 1, name = 'increment_global_step')

with tf.name_scope("embeddings"):
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size,
                           embedding_dimension],
                          -1.0, 1.0), name='embedding')
    embed = tf.nn.embedding_lookup(embeddings, _inputs)

with tf.variable_scope("lstm"):
    # Define a function that gives the output in the right shape
    def lstm_cell():
        return tf.nn.rnn_cell.LSTMCell(hidden_layer_size, forget_bias=1.0)
    cell = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell() for _ in range(num_LSTM_layers)],
                                       state_is_tuple=True)
    outputs, states = tf.nn.dynamic_rnn(cell, embed,
                                        sequence_length=_seqlens,
                                        dtype=tf.float32)
#### tags ####
with tf.name_scope("embeddings_tags"):
    embeddings_tags = tf.Variable(
        tf.random_uniform([vocabulary_size_tags,
                           embedding_dimension_tags],
                          -1.0, 1.0), name='embedding_tags')
    embed_tags = tf.nn.embedding_lookup(embeddings_tags, _inputs_tags)

with tf.variable_scope("lstm_tags"):
    # Define a function that gives the output in the right shape
    def lstm_cell():
        return tf.nn.rnn_cell.LSTMCell(hidden_layer_size_tags, forget_bias=1.0)
    cell_tags = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell() for _ in range(num_LSTM_layers_tags)],
                                       state_is_tuple=True)
    outputs_tags, states_tags = tf.nn.dynamic_rnn(cell_tags, embed_tags,
                                        sequence_length=_seqlens,
                                        dtype=tf.float32)
#######

weights = {
     'linear_layer': tf.Variable(tf.truncated_normal([hidden_layer_size + hidden_layer_size_tags, num_classes],
                                                     mean=0, stddev=.01))
 }
biases = {
    'linear_layer': tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=.01))
}
 # extract the last relevant output and use in a linear layer
lstm_states = tf.concat([states[num_LSTM_layers-1][1], states_tags[num_LSTM_layers_tags-1][1]], 1)
with tf.name_scope("Predictions"):
    final_output = tf.matmul(lstm_states, weights["linear_layer"]) + biases["linear_layer"]


with tf.name_scope("cost"):
    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output,
                                                      labels=_labels)
    cross_entropy = tf.reduce_mean(softmax)
    CE_summary = tf.summary.scalar("cross_entropy", cross_entropy)

train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(_labels, 1),
                              tf.argmax(final_output, 1))
with tf.name_scope("ACC"):
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100
    acc_summary = tf.summary.scalar("Accuracy", accuracy)

with tf.name_scope("Confusion_Matrix"):
    confusion = tf.confusion_matrix(labels=tf.argmax(_labels, 1),
                                    predictions=tf.argmax(final_output, 1),
                                    num_classes=num_classes)
    conf_summary = tf.summary.text("Confusion Matrix", tf.as_string(confusion))


with tf.name_scope("Missclassified"):
    sample = tf.where(tf.math.logical_not(correct_prediction))
    miss_summary = tf.summary.text("Miss Classification", _miss)

saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=0.5)


with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(os.path.join(SAVE_PATH + "/train"),
                                         graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(os.path.join(SAVE_PATH + "/test"),
                                         graph=tf.get_default_graph())
    with open(os.path.join(SAVE_PATH, 'metadata.tsv'), "w") as metadata:
        metadata.write('Name\tClass\n')
        for k, v in index2word_map.items():
            metadata.write('%s\t%d\n' % (v, k))
    if glob.glob(SAVE_PATH + '*.me