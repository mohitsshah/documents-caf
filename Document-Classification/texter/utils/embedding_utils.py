import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors
from keras.layers import Embedding

import warnings
warnings.simplefilter("ignore")

__all__ = ['glove_embedding_layer', 'word2vec_embedding_layer']


def glove_embedding_layer(pretrained_embeddings_path, word_index, MAX_WORDS=20000,
                          EMBEDDING_DIM=300, MAX_SEQUENCE_LENGTH=1000, pretrained_weights=True):
    """
    Returns a Keras Embedding layer, with pretrained glove embeddings.
    which can be found here: https://nlp.stanford.edu/projects/glove/

    Parameters:

    pretrained_embeddings_path: str
        path to pretrained glove embeddings.

    word_index: dict
        an index of words from the dataset, returned by the keras_tokenizer.

    MAX_WORDS: int
        max number of words to be included in the processing.

    EMBEDDING_DIM: int
        dimension of the embedding layer.

    MAX_SEQUENCE_LENGTH: int
        maximum length of the sequence to be considered.

    pretrained_weights: bool
        if `true` returns the layer with pretrainned embeddings, 
        if `false` returns a layer without pretrained embeddings.

    Returns: 

        Keras `Embedding` layer object


    """
    embeddings_index = {}
    with open(pretrained_embeddings_path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f'Found {len(embeddings_index)} word embeddings')

    vocabulary_size = min(MAX_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i >= MAX_WORDS:
            continue
        try:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = np.random.normal(
                0, np.sqrt(0.25), EMBEDDING_DIM)

    if pretrained_weights:
        emb_layer = Embedding(vocabulary_size, EMBEDDING_DIM, weights=[embedding_matrix],
                              input_length=MAX_SEQUENCE_LENGTH, trainable=True)

    emb_layer = Embedding(vocabulary_size, EMBEDDING_DIM)

    return emb_layer


def word2vec_embedding_layer(pretrained_embeddings_path, word_index, NUM_WORDS=20000,
                             EMBEDDING_DIM=300, pretrained_weights=True):
    """
    Returns a Keras Embedding layer, with pretrained word2vec embeddings.
    which can be found here: 
    https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

    Parameters:

    pretrained_embeddings_path: str
        path to pretrained gword2vec embeddings.

    word_index: dict
        an index of words from the dataset, returned by the keras_tokenizer.

    MAX_WORDS: int
        max number of words to be included in the processing.

    EMBEDDING_DIM: int
        dimension of the embedding layer.

    pretrained_weights: bool
        if `true` returns the layer with pretrainned embeddings, 
        if `false` returns a layer without pretrained embeddings.

    Returns: 

        Keras `Embedding` layer object

    """
    word_vectors = KeyedVectors.load_word2vec_format(
        pretrained_embeddings_path, binary=True)
    EMBEDDING_DIM = EMBEDDING_DIM
    vocabulary_size = min(len(word_index)+1, NUM_WORDS)
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= NUM_WORDS:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = np.random.normal(
                0, np.sqrt(0.25), EMBEDDING_DIM)

    if pretrained_weights:
        emb_layer = Embedding(vocabulary_size, EMBEDDING_DIM, weights=[
                              embedding_matrix], trainable=True)
    emb_layer = Embedding(vocabulary_size, EMBEDDING_DIM)

    return emb_layer
