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

np.random.seed(15101993)


def vanilla_cnn(max_seq_len, n_class, embedding_layer, activation='relu',
                loss="categorical_crossentropy",
                optimizer='adam', metrics=['acc'],
                batch_shape=None, sparse=False):

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
