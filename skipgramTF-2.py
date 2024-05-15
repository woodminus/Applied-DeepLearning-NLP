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
            index = 0  # dictionar