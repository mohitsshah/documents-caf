import numpy as np

import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors

from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import regularizers
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout, concatenate
from keras.layers.core import Reshape, Flatten


np.random.seed(15101993)


def kim_cnn(sequence_length, EMBEDDING_DIM, embedding_layer, n_class, lr=1e-3, beta_1=0.9,
            beta_2=0.999, epsilon=None, decay=0.0,
            drop=0.5, num_filters=100, filter_sizes=[3, 4, 5],
            loss='categorical_crossentropy', metrics=['acc']):

    inputs = Input(shape=(sequence_length,))
    embedding = embedding_layer(inputs)
    reshape = Reshape((sequence_length, EMBEDDING_DIM, 1))(embedding)

    conv_0 = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),
                    activation='relu', kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_1 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),
                    activation='relu', kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),
                    activation='relu', kernel_regularizer=regularizers.l2(0.01))(reshape)

    maxpool_0 = MaxPooling2D(
        (sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1))(conv_0)
    maxpool_1 = MaxPooling2D(
        (sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1))(conv_1)
    maxpool_2 = MaxPooling2D(
        (sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1))(conv_2)

    merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
    flatten = Flatten()(merged_tensor)
    reshape = Reshape((3*num_filters,))(flatten)  # 3: replace and try, a different number?
    dropout = Dropout(drop)(flatten)
    output = Dense(units=n_class, activation='softmax',
                   kernel_regularizer=regularizers.l2(0.01))(dropout)

    model = Model(inputs, output)

    adam = Adam(lr=lr, beta_1=beta_1,
                beta_2=beta_2, epsilon=epsilon, decay=decay)

    model.compile(loss=loss, optimizer=adam, metrics=metrics)

    return model
