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

examples = tfds.load(name='wmt14_translate/hi-en',
          as_supervised=True,
          with_info=True, 
          download=True)
examples = examples[0]
training_examples = examples['train']

for de, en in training_examples.take(2):
    print("German:", de.numpy().decode('utf-8'))
    print("English:", en.numpy().decode('utf-8'))

train_en = training_examples.map(lambda pt, en: en)
train_de = training_examples.map(lambda de, en: de)
#tokenizer = text.BertTokenizer("vocab.txt", token_out_type=tf.string)
#tokenizer = text.WhitespaceTokenizer()
#tokenized_docs = train_de.map(lambda x: tokenizer.tokenize(x))
#iterator = iter(tokenized_docs)
#print(next(iterator))
bert_tokenizer_params = dict(lower_case=False)
reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]

bert_vocab_args = dict(
    vocab_size = 800000,
    reserved_tokens=reserved_tokens,
    bert_tokenizer_params=bert_tokenizer_params,
    learn_params={},
)
print(type(train_de))
de_vocab = bert_vocab.bert_vocab_from_dataset(
        train_de.batch(100000).prefetch(2000),
        **bert_vocab_args
)


print(de_vocab[:10])
print(de_vocab[100:110])
print(de_vocab[1000:1010])
print(de_vocab[-10:])

def write_vocab_file(filepath, vocab):
  with open(filepath, 'w') as f:
    for token in vocab:
      print(token, file=f)

write_vocab_file('hi.txt', de_vocab)

##train_examples, val_examples = examples['train'], examples['validation']
##for pt, en in train_examples.take(1):
##  print("Portuguese: ", pt.numpy().decode('utf-8'))
##  print("English:   ", en.numpy().decode('utf-8'))  
