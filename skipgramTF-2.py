# based in parts on https://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/
# and https://adventuresinmachinelearning.com/word2vec-keras-tutorial/
# rewritten for tensorflow 2.2 integrated subword tokenizer

import tensorflow as tf
import sentencepiece as spm
import collections
import numpy as np
import io

# parameters
filename = "/home/simon/Downloads/poems_processed.txt" # Any big txt...
vocab_size=10000
sub_size=15000
window_size = 3
vector_dim = 768
epochs = 1000000

valid_size = 8     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

with open(filename, "r", encoding = "utf8") as f: text = f.read()

# functions

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


spm.SentencePieceTrainer.Train('--input={} --model_prefix=m --vocab_size={}'.format(filename, sub_size))

sp = spm.SentencePieceProcessor()
sp.load('m.model')

print(sp.encode_as_pieces('this is a test.'))

vocabulary = sp.encode_as_pieces(text)

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocab_size)


sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
couples, labels = tf.keras.preprocessing.sequence.skipgrams(data, vocab_size, window_size=window_size, sampling_table=sampling_table)
word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")

print(couples[:10], labels[:10])

# create some input variables
input_target = tf.keras.Input((1,))
input_context = tf.keras.Input((1,))

embedding = tf.keras.layers.Embedding(vocab_size, vector_dim, input_length=1, name='embedding')

target = embedding(input_target)
target = tf.keras.layers.Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = tf.keras.layers.R