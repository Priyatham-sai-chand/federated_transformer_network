import collections
import logging
import os
import pathlib
import re
import string
import sys
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

logging.getLogger('tensorflow').setLevel('WARNING')  # suppress warnings

data_dir = 'D:\\tensorflow_datasets'
examples = tfds.load(name='wmt14_translate/de-en',
          data_dir=data_dir, 
          as_supervised=True,
          with_info=True, 
          download=False)
examples = examples[0]
training_examples = examples['train']

for de, en in training_examples.take(1):
    print("German:", de.numpy().decode('utf-8'))
    print("English:", en.numpy().decode('utf-8'))

bert_tokenizer_params = dict(lower_case=False)
reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]

bert_vocab_args = dict(
    vocab_size = 16000,
    reserved_tokens=reserved_tokens,
    bert_tokenizer_params=bert_tokenizer_params,
    learn_params={},
)

train_en = training_examples.map(lambda pt, en: en)
train_de = training_examples.map(lambda de, en: de)

de_tokenizer = text.BertTokenizer('de_vocab.txt', **bert_tokenizer_params)
en_tokenizer = text.BertTokenizer('en_vocab.txt', **bert_tokenizer_params)

# Tokenize the examples -> (batch, word, word-piece)
token_batch = en_tokenizer.tokenize(train_en)
# Merge the word and word-piece axes -> (batch, tokens)
token_batch = token_batch.merge_dims(-2,-1)

for ex in token_batch.to_list():
  print(ex)

