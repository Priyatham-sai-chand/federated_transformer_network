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
training_examples = examples['validation']

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
        train_de.batch(1000).prefetch(2),
        **bert_vocab_args
)

en_vocab = bert_vocab.bert_vocab_from_dataset(
        train_en.batch(1000).prefetch(2),
        **bert_vocab_args
)

print(de_vocab[1000:1010])
print(en_vocab[1000:1010])

def write_vocab_file(filepath, vocab):
  with open(filepath, 'w', encoding="utf-8") as f:
    for token in vocab:
      print(token, file=f)

write_vocab_file('de_vocab.txt', de_vocab)
write_vocab_file('en_vocab.txt', en_vocab)

de_tokenizer = text.BertTokenizer('de_vocab.txt', **bert_tokenizer_params)
en_tokenizer = text.BertTokenizer('en_vocab.txt', **bert_tokenizer_params)


START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
END = tf.argmax(tf.constant(reserved_tokens) == "[END]")

def add_start_end(ragged):
  count = ragged.bounding_shape()[0]
  starts = tf.fill([count,1], START)
  ends = tf.fill([count,1], END)
  return tf.concat([starts, ragged, ends], axis=1)


def cleanup_text(reserved_tokens, token_txt):
  # Drop the reserved tokens, except for "[UNK]".
  bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
  bad_token_re = "|".join(bad_tokens)

  bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
  result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

  # Join them into strings.
  result = tf.strings.reduce_join(result, separator=' ', axis=-1)

  return result


class CustomTokenizer(tf.Module):
  def __init__(self, reserved_tokens, vocab_path):
    self.tokenizer = text.BertTokenizer(vocab_path, lower_case=False)
    self._reserved_tokens = reserved_tokens
    self._vocab_path = tf.saved_model.Asset(vocab_path)

    vocab = pathlib.Path(vocab_path).read_text('utf-8').splitlines()
    self.vocab = tf.Variable(vocab)

    ## Create the signatures for export:   

    # Include a tokenize signature for a batch of strings. 
    self.tokenize.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string))

    # Include `detokenize` and `lookup` signatures for:
    #   * `Tensors` with shapes [tokens] and [batch, tokens]
    #   * `RaggedTensors` with shape [batch, tokens]
    self.detokenize.get_concrete_function(
        tf.TensorSpec(shape=[None, None], dtype=tf.int64))
    self.detokenize.get_concrete_function(
          tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

    self.lookup.get_concrete_function(
        tf.TensorSpec(shape=[None, None], dtype=tf.int64))
    self.lookup.get_concrete_function(
          tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

    # These `get_*` methods take no arguments
    self.get_vocab_size.get_concrete_function()
    self.get_vocab_path.get_concrete_function()
    self.get_reserved_tokens.get_concrete_function()

  @tf.function
  def tokenize(self, strings):
    enc = self.tokenizer.tokenize(strings)
    # Merge the `word` and `word-piece` axes.
    enc = enc.merge_dims(-2,-1)
    enc = add_start_end(enc)
    return enc

  @tf.function
  def detokenize(self, tokenized):
    words = self.tokenizer.detokenize(tokenized)
    return cleanup_text(self._reserved_tokens, words)

  @tf.function
  def lookup(self, token_ids):
    return tf.gather(self.vocab, token_ids)

  @tf.function
  def get_vocab_size(self):
    return tf.shape(self.vocab)[0]

  @tf.function
  def get_vocab_path(self):
    return self._vocab_path

  @tf.function
  def get_reserved_tokens(self):
    return tf.constant(self._reserved_tokens)


tokenizers = tf.Module()
tokenizers.de = CustomTokenizer(reserved_tokens, 'de_vocab.txt')
tokenizers.en = CustomTokenizer(reserved_tokens, 'en_vocab.txt')


model_name = 'ttranslate_de_en_converter'
tf.saved_model.save(tokenizers, model_name)



