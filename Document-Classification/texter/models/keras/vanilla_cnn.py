import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.callbacks import EarlyStopping

import warnings
warnings.simplefilter("ignore")

np.random.seed(15101993)


def vanilla_cnn(max_seq_len, n_class, embedding_layer, activation='relu',
                loss="categorical_crossentropy",
                optimizer='adam', metrics=['acc'],
                batch_shape=None, sparse=False):
    """
    returns a compiled keras Vanilla CNN model

    Parameters:

    max_seq_len: int
        length of the sentence/sequence

    embedding_layer: keras object
        keras embedding layer object,
        returned from embedding_utils module

    activation: str
        type of activation function to be used.
         for a complete list please check keras documentation.

    n_class: int
        number of classification categories.

    loss: str
        type of loss function to be used

    optimizer: str
        type of optimizer function to be used. 
        for a complete list please check keras documentation.

    metrics: list
        types of evaluation metrics to be considered, 
        for the complete list of accepted values check keras documentation.

    batch_shape: optional/int
        bastch shape of the data to be used

    sparse: optional/bool
        specifies if the data is sparsely shaped or not.

    Returns: 

        compiled keras model object

    """

    sequence_input = Input(shape=(max_seq_len,), dtype='int32',
                           batch_shape=batch_shape, sparse=sparse)
    embedded_sequences = embedding_layer(sequence_input)

    x = Conv1D(256, 5, activation=activation)(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(64, 5, activation=activation)(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(256, 5, activation=activation)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation=activation)(x)
    preds = Dense(units=n_class, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
