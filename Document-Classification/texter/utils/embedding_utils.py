import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors
from keras.layers import Embedding


def glove_embedding_layer(pretrained_embeddings_path, word_index, MAX_WORDS=20000,
                          EMBEDDING_DIM=300, MAX_SEQUENCE_LENGTH=1000, pretrained_weights=True):

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
