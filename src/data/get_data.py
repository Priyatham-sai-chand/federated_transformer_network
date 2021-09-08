import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


data_dir = 'D:\\tensorflow_datasets'

examples = tfds.load(name='wmt16_translate/de-en',
          data_dir=data_dir, 
          as_supervised=True,
          with_info=True, 
          download=False)
print(type(examples))
print(type(examples[0]))
examples = examples[0]
training_examples = examples['train']

for de, en in training_examples.take(2):
  print("German", de.numpy().decode('utf-8'))
  print("English:   ", en.numpy().decode('utf-8'))

train_en = training_examples.map(lambda pt, en: en)
train_de = training_examples.map(lambda de, en: de)
reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]

bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size = 1000,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=reserved_tokens,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params={},
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)
print(type(train_de))
print(train_en)
de_vocab = bert_vocab.bert_vocab_from_dataset(
        train_de.batch(1000).prefetch(2),
        **bert_vocab_args
)


print(de_vocab[:10])
print(de_vocab[100:110])
print(de_vocab[1000:1010])
print(de_vocab[-10:])



##train_examples, val_examples = examples['train'], examples['validation']
##for pt, en in train_examples.take(1):
##  print("Portuguese: ", pt.numpy().decode('utf-8'))
##  print("English:   ", en.numpy().decode('utf-8'))  
