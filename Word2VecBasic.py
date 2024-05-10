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
default_