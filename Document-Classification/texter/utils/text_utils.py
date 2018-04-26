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

__all__ = ['accent_cleaner', 'whitespace_normalizer', 'replace_numbers',
           'fix_bad_unicode', 'train_test_split_data', 'vectorizer',  'vectorized_data',  'sklearn_text_vectorizer', 'sklearn_text_vectorizer', 'pos_tagger', 'document_pos_tagger', 'word2vec', 'keras_text_tokenizer']

def accent_cleaner(text, mode='unicode'):
    """
    removes the accent from the texts

    Parameters:

    text: str
        raw text

    mode: str 
    either 'unicode' or 'ascii'. if ‘unicode’, remove accented char for any unicode symbol with a direct ASCII equivalent; if ‘ascii’, remove accented char for any unicode symbol.

    Returns:	
        str

    """
    if mode == 'unicode':
        return ''.join(c for c in unicodedata.normalize('NFKD', text)
                       if not unicodedata.combining(c))
    elif mode == 'ascii':
        return unicodedata.normalize('NFKD', text).encode('ascii', errors='ignore').decode('ascii')
    else:
        msg = f'`mode` must be  "unicode" or "ascii", not {mode}'
        raise ValueError(msg)


def whitespace_normalizer(text):
    """removes the whitespace from the text file

    Parameters: 

    text: str
        text file

    Returns: 
        str, text file

    """
    linebreaks = re.compile(r'((\r\n)|[\n\v])+')
    nonbreak_space = re.compile(r'(?!\n)\s+')
    return nonbreak_space.sub(' ', linebreaks.sub(r'\n', text)).strip()


def replace_numbers(text, replace_with='*Number*'):
    """removes the numbers from the text file and replaces with a custom text.

    Parameters: 

    text: str
        text file

    replace_with: str
        replace the number with this text

    Returns: 
        str, text file

    """
    numbers = re.compile(
        r'(?:^|(?<=[^\w,.]))[+–-]?(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)(?:$|(?=\b))')
    return numbers.sub(replace_with, text)


def fix_bad_unicode(text, normalization='NFC'):
    """removes the whitespace from the text file

    Parameters: 

    text: str
        text file

    normalization: str
        type pf normalization to be done, choose one of 'NFC', 'NFKC', 'NFD', 'NFKD'.

    Returns: 
        str, text file

    """
    return fix_text(text, normalization=normalization)


def train_test_split_data(texts, labels, test_size=0.2):
    """
    prepares the train test data of given test size param.

    Parameters:

    texts: np.array
        text files

    labels: np.array
        class labels of the text files.

    test_size: float
        size of the test split

    Returns: np.array
        train/test labels & texts 

    """
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
    """
    utility function to create the vector representation 
    of the input data files using tfidf sklearn model. 
    this function needs to be used when you want to train 
    a sklearn based classifier.

    Parameters:

    train_texts: numpy array
        training documents

    test_texts: numpy array
        test documents

    encoding: str
        type of encoding

    decode_error: str
        how to handle the decoding error, choose one of these: 'strict', 'ignore', 'replace'

    analyzer : string, {'word', 'char'} or callable
        Whether the feature should be made of word or character n-grams. If a callable is passed it is used to extract the sequence of features out of the raw, unprocessed input.  

    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while preserving the tokenizing and n-grams generation steps.  

    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the preprocessing and n-grams generation steps. Only applies if `analyzer == 'word'`.  

    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n  = n  = max_n will be used.  

    stop_words : string {'english'}, list, or None (default)
        If a string, it is passed to _check_stop_list and the appropriate stop list is returned. 'english' is currently the only supported string value. If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.Only applies if `analyzer == 'word'`.  

        If None, no stop words will be used. max_df can be set to a value in the range [0.7, 1.0)to automatically detect and filter stop words based on intra corpus document frequency of terms.  

    token_pattern : string
        Regular expression denoting what constitutes a "token", only used if `analyzer == 'word'`. The default regexp selects tokens of 2 or more alphanumeric characters (punctuation is completely ignored and always treated as a token separator).  

    max_df : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).If float, the parameter represents a proportion of documents, integer absolute counts.This parameter is ignored if vocabulary is not None.  

    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None. 

    max_features : int or None, default=None 
        If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. This parameter is ignored if vocabulary is not None. 

    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are indices in the feature matrix, or an iterable over terms. If not given, a vocabulary is determined from the input documents.  

    binary : boolean, default=False
        If True, all non-zero term counts are set to 1. This does not mean outputs will have only 0/1 values, only that the tf term in tf-idf is binary. (Set idf and normalization to False to get 0/1 outputs.)

    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().  

    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.  

    use_idf : boolean, default=True
        Enable inverse-document-frequency reweighting.  

    smooth_idf : boolean, default=True
        Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions.  

    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    Returns: 
        train_data: np.array
            training data in vectorized format
        test_data: np.array
            testing data in vectorized format
        fitted_model: sklearn object
            text vectorization model
    """

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
    """
    utility function to create the vector representation of the input data 
    files using tfidf sklearn model. this function needs to be used 
    when you want to train a sklearn based classifier.
    note: this is a util function that wraps around `vectorizer`
     & `train_test_split_data` functions. 

    Parameters:

    texts: numpy array
        text documents

    labels: numpy array
        class labels of the text documents

    test_size: float
        test data split size

    encoding: str
        type of encoding

    decode_error: str
        how to handle the decoding error, choose one of these: 'strict', 'ignore', 'replace'

    analyzer : string, {'word', 'char'} or callable
        Whether the feature should be made of word or character n-grams. If a callable is passed it is used to extract the sequence of features out of the raw, unprocessed input.  

    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while preserving the tokenizing and n-grams generation steps.  

    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the preprocessing and n-grams generation steps. Only applies if `analyzer == 'word'`.  

    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n  = n  = max_n will be used.  

    stop_words : string {'english'}, list, or None (default)
        If a string, it is passed to _check_stop_list and the appropriate stop list is returned. 'english' is currently the only supported string value. If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.Only applies if `analyzer == 'word'`.  

        If None, no stop words will be used. max_df can be set to a value in the range [0.7, 1.0)to automatically detect and filter stop words based on intra corpus document frequency of terms.  

    token_pattern : string
        Regular expression denoting what constitutes a "token", only used if `analyzer == 'word'`. The default regexp selects tokens of 2 or more alphanumeric characters (punctuation is completely ignored and always treated as a token separator).  

    max_df : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).If float, the parameter represents a proportion of documents, integer absolute counts.This parameter is ignored if vocabulary is not None.  

    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None. 

    max_features : int or None, default=None 
        If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. This parameter is ignored if vocabulary is not None. 

    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are indices in the feature matrix, or an iterable over terms. If not given, a vocabulary is determined from the input documents.  

    binary : boolean, default=False
        If True, all non-zero term counts are set to 1. This does not mean outputs will have only 0/1 values, only that the tf term in tf-idf is binary. (Set idf and normalization to False to get 0/1 outputs.)

    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().  

    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.  

    use_idf : boolean, default=True
        Enable inverse-document-frequency reweighting.  

    smooth_idf : boolean, default=True
        Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions.  

    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    Returns: 
        train_data: np.array
            training data in vectorized format
        test_data: np.array
            testing data in vectorized format
        train_labels: np.array
            training data in labels
        test_labels: np.array
            testing data in labels
        fitted_model: sklearn object
            text vectorization model
    """
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


def sklearn_text_vectorizer(text,  fitted_model=None, w2v_path=None, model="tfidf",
                            remove_pattern="[^a-zA-Z]"):
    """utility function to vectorize the texts while predicting on the incoming unknown text files

    Parameters: 

    text: str
        new unknown text document

    w2v_path: None/str
        path to word embeddings

    fitted_model: sklearn object/None
        vectorization model used to vectorize the training dataset

    model: str
        choose either tfidf or word2vec

    remove_pattern: str
        regex pattern to remove unwanted chars from the text

    Returns:
        vectorized text

    """
    if model == "tfidf":
        excludes = stopwords.words("english")
        pattern = re.compile(remove_pattern)
        content = pattern.sub(" ", text)
        words = content.split()
        words = [w for w in words if len(w) > 1]
        words = [w for w in words if w not in excludes]
        if len(words) > 0:
            lines = ' '.join(words)
        return fitted_model.transform((lines, ))

    if model == "word2vec":
        return word2vec([pos_tagger(text)], w2v_path)


def keras_text_vectorizer(text, tokenizer, MAXLEN=2000):
    """
    utility function to vectorize the texts while predicting on the incoming unknown text files

    Parameters: 

    text: str
        new unknown text document

    tokenizer: keras object
        keras tokenizer object generated by keras_data_config function

    MAXLEN: int
        maximum length of the sequence used, must be same as the input data length.

    Returns: numpy array
        vectorized text data which can be used for predictions.

    """
    sample = [text]
    tokenizer.fit_on_texts(sample)
    return pad_sequences(tokenizer.texts_to_sequences(sample), MAXLEN)


def keras_text_tokenizer(x_train, x_test, y_train, y_test, num_class, MAXLEN=2000,
                         NUM_WORDS=20000, lower=True):
    """
    keras text tokenization utility

    Parameters: 

    x_train: np.array
        training data

    x_test: np.array
        testing data

    y_train: np.array
        training labels

    y_test: np.array
        testing labels

    num_class: int
        number of classes/categories

    MAXLEN: int 
        maximum sequence length

    NUM_WORDS: int
        maximum number of words to be used in the input representation.

    lower: bool
        Whether to convert the texts to lowercase or not.

    Returns: 

    X_train: np.array
        vectorized input train docs

    X_val: np.array
        vectorized input test docs 

    Y_train: np.array 
        categorical train labels 

    Y_val: np.array
        categorical test labels

    tokenizer: keras object
        tokenizer model

    word_index: dict
        dictionary of input text    
    """
    tokenizer = Tokenizer(num_words=NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                          lower=lower)
    tokenizer.fit_on_texts(x_train)
    sequences_train = tokenizer.texts_to_sequences(x_train)
    sequences_valid = tokenizer.texts_to_sequences(x_test)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    X_train = pad_sequences(sequences_train, maxlen=MAXLEN)
    X_val = pad_sequences(sequences_valid, maxlen=MAXLEN)
    Y_train = to_categorical(y_train, num_class)
    Y_val = to_categorical(y_test, num_class)
    print('Shape of X train and X validation tensor:', X_train.shape, X_val.shape)
    print('Shape of label train and validation tensor:',
          Y_train.shape, Y_val.shape)
    return X_train, X_val, Y_train, Y_val, tokenizer, word_index


def pos_tagger(data):
    """
    does pos tagging on the input text file

    Parameters: 

    data: str
        text file

    Returns: list
        pos tagged list of words
    """
    sentences = sent_tokenize(data)
    sents = []
    for s in sentences:
        text = word_tokenize(s)
        pos_tagged = pos_tag(text)
        sub_tags = [
            (word, map_tag('en-ptb', 'universal', tag)) for word, tag in pos_tagged]
        sents.append(sub_tags)
    return sents

# TODO: remove pandas dependency


def document_pos_tagger(data_df):
    """
    does pos tagging on the input text documents

    Parameters: 

    data_df: dataframe
        text file

    Returns: 

    x: list
        pos tagged list of sentences

    y: list
        labels 

    """
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
    """
    creates a word2vec model from the training dataset.

    Parameters: 

    x: list
        list of texts

    w2v_path: str
        path to the word2vec pretrained model

    limit: int
        max number of words

    pos_filter: list
        list of pos tags to keep in the output embeddings

    Returns:

        numpy array of embeddings of the original text files.


    """
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
