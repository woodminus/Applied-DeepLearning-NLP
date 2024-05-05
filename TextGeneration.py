
# -*- coding: utf-8 -*-


# https://gist.github.com/mikalv/3947ccf21366669ac06a01f39d7cff05
# http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
import tensorflow as tf
import numpy as np
import os, sys
import re
import collections

#set hyperparameters
max_len = 40
step = 10
num_units = 128
learning_rate = 0.001
batch_size = 200
epoch = 50
temperature = 0.8
SAVE_PATH = 'logs/TextGen/Rilke/'


if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

def tokens(text):
    """
    Get all words from corpus
    """
    text = re.sub(r'[0-9]+', '', text)
    return re.findall(r'\w+', text.lower())

WORDS = tokens(file('RilkeBig.txt').read())
WORD_COUNTS = collections.Counter(WORDS)

def edits0(word):
    """
    Return all strings that are zero edits away (i.e. the word itself).
    """
    return{word}

def edits1(word):
    """
    Return all strings that are one edits away.
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyzäüö'
    def splits(word):
        """
        return a list of all possible pairs
        that the input word is made of
        """
        return [(word[:i], word[i:]) for i in range(len(word)+1)]
    pairs = splits(word)
    deletes = [a+b[1:] for (a,b) in pairs if b]
    transposes = [a+b[1]+b[0]+b[2:] for (a,b) in pairs if len(b) >1]
    replaces = [a+c+b[1:] for (a,b) in pairs for c in alphabet if b]
    inserts = [a+c+b for (a,b) in pairs for c in alphabet]
    return(set(deletes + transposes + replaces + inserts))

def edits2(word):
    """
    return all strings that are two edits away.
    """
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}

def known(words):
    return {w for w in words if w in WORD_COUNTS}

def correct(word):
    candidates = (known(edits0(word)) or
                 known(edits1(word)) or
                 known(edits2(word)) or
                 [word])
    return max(candidates, key=WORD_COUNTS.get)

def correct_match(match):#
    """
    spell-correct word in match,
    and perserve upper/lower/title case
    """
    word = match.group()
    def case_of(text):
        return(str.upper if text.isupper() else
              str.lower if text.islower() else
              str.title if text.istitle() else
              str)
    return case_of(word)(correct(word.lower()))

def correct_text_generic(text):
    """
    correct all words in text
    """
    return re.sub('[a-zA-Z]+', correct_match, text)

def read_data(file_name):
    '''
     open and read text file
    '''
    text = open(file_name, 'r').read()
    return text.lower()

def featurize(text):
    '''
     featurize the text to train and target dataset
    '''
    unique_chars = list(set(text))
    len_unique_chars = len(unique_chars)

    input_chars = []
    output_char = []

    for i in range(0, len(text) - max_len, step):
        input_chars.append(text[i:i+max_len])
        output_char.append(text[i+max_len])

    train_data = np.zeros((len(input_chars), max_len, len_unique_chars))
    target_data = np.zeros((len(input_chars), len_unique_chars))

    for i , each in enumerate(input_chars):
        for j, char in enumerate(each):
            train_data[i, j, unique_chars.index(char)] = 1
        target_data[i, unique_chars.index(output_char[i])] = 1
    return train_data, target_data, unique_chars, len_unique_chars

def rnn(x, weight, bias, len_unique_chars):
    '''
     define rnn cell and prediction
    '''
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, len_unique_chars])
    x = tf.split(x, max_len, 0)