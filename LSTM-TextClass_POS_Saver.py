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
    d_tuples = [nltk.pos_t