import os
import urllib
import tensorflow as tf
import zipfile
import collections
import numpy as np
import io

def maybe_download(filename, url, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

url = 'http://mattmahoney.net/dc/'
filename = maybe_download('text8.zip', url, 31344016)

# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

vocabulary = read_data(filename)
print(vocabulary[:7])

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

vocab_size=10000
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocab_size)

window_size = 3
vector_dim = 3
epochs = 100

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

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
context = tf.keras.layers.Reshape((vector_dim, 1))(context)


similarity = tf.keras.layers.Dot(normalize=True, axes=1)([target, context])

# now perform the dot product operation to get a similarity measure
dot_product = tf.keras.layers.Dot(axes=1)([target, context])
dot_product = tf.keras.layers.Reshape((1,))(dot_product)
# add the sigmoid output layer
output = tf.kera