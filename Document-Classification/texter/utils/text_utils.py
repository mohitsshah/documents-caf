import re
import os
import glob
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk import sent_tokenize
from nltk import pos_tag
from nltk import map_tag
from nltk import word_tokenize

from gensim.models import KeyedVectors
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import unicodedata
from ftfy import fix_text
from unidecode import unidecode

import warnings
warnings.simplefilter("ignore")

__all__ = ['load_data', 'accent_cleaner', 'whitespace_normalizer', 'replace_numbers',
           'fix_bad_unicode', 'train_test_split_data', 'vectorizer',  'vectorized_data',  'text_vectorizer', 'pos_tagger', 'document_pos_tagger', 'word2vec']


def load_data(exclude, text_directory_path, class_labels, remove_pattern="[^a-zA-Z]"):
    excludes = exclude
    pattern = re.compile(remove_pattern)
    texts = []
    labels = []
    textdir = text_directory_path
    categories = class_labels
    for i, category in enumerate(categories):
        path = os.path.join(textdir, category)
        files = os.listdir(path)
        for f in files:
            p = os.path.join(path, f)
            content = open(p, 'r').read()
            content = pattern.sub(" ", content)
            words = content.split()
            words = [w for w in words if len(w) > 1]
            words = [w for w in words if w not in excludes]
            if len(words) > 0:
                lines = ' '.join(words)
                texts.append(lines)
                labels.append(i)
    return pd.DataFrame(dict(text=texts, label=labels))


def accent_cleaner(text, mode='unicode'):
    if mode == 'unicode':
        return ''.join(c for c in unicodedata.normalize('NFKD', text)
                       if not unicodedata.combining(c))
    elif mode == 'ascii':
        return unicodedata.normalize('NFKD', text).encode('ascii', errors='ignore').decode('ascii')
    else:
        msg = f'`mode` must be  "unicode" or "ascii", not {mode}'
        raise ValueError(msg)


def whitespace_normalizer(text):
    linebreaks = re.compile(r'((\r\n)|[\n\v])+')
    nonbreak_space = re.compile(r'(?!\n)\s+')
    return nonbreak_space.sub(' ', linebreaks.sub(r'\n', text)).strip()


def replace_numbers(text, replace_with='*Number*'):
    numbers = re.compile(
        r'(?:^|(?<=[^\w,.]))[+–-]?(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)(?:$|(?=\b))')
    return numbers.sub(replace_with, text)


def fix_bad_unicode(text, normalization='NFC'):
    return fix_text(text, normalization=normalization)


def train_test_split_data(texts, labels, test_size=0.2):
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size)
    return train_texts, test_texts, train_labels, test_labels


def vectorizer(train_texts, test_texts, encoding='utf-8',
               decode_error='strict', preprocessor=None,
               tokenizer=None, analyzer='word', stop_words=None,
               token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1),
               max_df=1.0, min_df=1, max_features=None, vocabulary=None,
               binary=False, norm='l2', use_idf=True, smooth_idf=True,
               sublinear_tf=False):

    V = TfidfVectorizer(encoding=encoding, decode_error=decode_error, preprocessor=preprocessor,
                        tokenizer=tokenizer, analyzer=analyzer, stop_words=stop_words,
                        token_pattern=token_pattern, ngram_range=ngram_range, max_df=max_df,
                        min_df=min_df, max_features=max_features, vocabulary=vocabulary,
                        binary=binary, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
                        sublinear_tf=sublinear_tf)

    fitted_model = V.fit(train_texts)
    train_data = fitted_model.transform(train_texts)
    test_data = fitted_model.transform(test_texts)

    return train_data, test_data, fitted_model


def vectorized_data(texts, labels, test_size=0.2, encoding='utf-8',
                    decode_error='strict', preprocessor=None,
                    tokenizer=None, analyzer='word', stop_words=None,
                    token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1),
                    max_df=1.0, min_df=1, max_features=None, vocabulary=None,
                    binary=False, norm='l2', use_idf=True, smooth_idf=True,
                    sublinear_tf=False):
    train_texts, test_texts, train_labels, test_labels = train_test_split_data(texts=texts,
                                                                               labels=labels,
                                                                               test_size=test_size)

    train_data, test_data, fitted_model = vectorizer(train_texts, test_texts, encoding=encoding, decode_error=decode_error, preprocessor=preprocessor,
                                                     tokenizer=tokenizer, analyzer=analyzer, stop_words=stop_words,
                                                     token_pattern=token_pattern, ngram_range=ngram_range, max_df=max_df,
                                                     min_df=min_df, max_features=max_features, vocabulary=vocabulary,
                                                     binary=binary, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
                                                     sublinear_tf=sublinear_tf)

    return train_data, test_data, train_labels, test_labels, fitted_model


def text_vectorizer(text, exclude, fitted_model, remove_pattern="[^a-zA-Z]"):
    excludes = exclude
    pattern = re.compile(remove_pattern)
    content = pattern.sub(" ", text)
    words = content.split()
    words = [w for w in words if len(w) > 1]
    words = [w for w in words if w not in excludes]
    if len(words) > 0:
        lines = ' '.join(words)
    return fitted_model.transform((lines, ))


def keras_tokenizer(max_n_words, texts, labels, max_seq_len):
    tokenizer = Tokenizer(num_words=max_n_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print(f'Found {len(word_index)} unique tokens.')
    data = pad_sequences(sequences, maxlen=max_seq_len)
    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    return data, labels, tokenizer, sequences, word_index


def keras_train_test_split(data, labels, val_split):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(val_split * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    return x_train, x_val, y_train, y_val


def pos_tagger(data):
    sentences = sent_tokenize(data)
    sents = []
    for s in sentences:
        text = word_tokenize(s)
        pos_tagged = pos_tag(text)
        sub_tags = [
            (word, map_tag('en-ptb', 'universal', tag)) for word, tag in pos_tagged]
        sents.append(sub_tags)
    return sents


def document_pos_tagger(data_df):
    x = []
    y = []
    total = len(data_df.text.as_matrix().tolist())
    texts = data_df.text.as_matrix().tolist()
    labels = data_df.label.as_matrix()
    for i in range(len(texts)):
        sents = pos_tagger(texts[i])
        x.append(sents)
        y.append(labels[i])
    return x, y


def word2vec(x, w2v_path, limit=200000, pos_filter=['ADJ', 'NOUN']):
    print("Loading Slim-Google-vectors-negative300.bin pretrained embeddings")
    google_vecs = KeyedVectors.load_word2vec_format(
        w2v_path, binary=True, limit=limit)
    print(f"Considering only {pos_filter}")
    print("Averaging the Word Embeddings...")
    x_embeddings = []
    total = len(x)
    processed = 0
    for tagged_text in x:
        count = 0
        doc_vector = np.zeros(300)
        for sentence in tagged_text:
            for tagged_word in sentence:
                if tagged_word[1] in pos_filter:
                    try:
                        doc_vector += google_vecs[tagged_word[0]]
                        count += 1
                    except KeyError:
                        continue
        doc_vector /= count
        if np.isnan(np.min(doc_vector)):
            continue

        x_embeddings.append(doc_vector)

        processed += 1
        if processed % 10000 == 0:
            print(processed, "/", total)

    return np.array(x_embeddings)
