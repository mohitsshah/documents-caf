from .text_utils import train_test_split_data, keras_text_tokenizer
from .text_utils import vectorized_data, document_pos_tagger
from .text_utils import word2vec

from .io_utils import load_data
from .io_utils import load_config, save_config
from .io_utils import load_model, save_model
from .io_utils import load_text_model, save_text_model

from .embedding_utils import glove_embedding_layer, word2vec_embedding_layer

from ..models.keras.kim_cnn import kim_cnn
from ..models.keras.vanilla_cnn import vanilla_cnn
from ..models.sklearn.learners import classifier

from keras.models import model_from_json
from sklearn.externals import joblib
import numpy as np
import json

__all__ = ['keras_data_config', 'keras_model_config',
           'sklearn_data_config', 'sklearn_model_config']

# TODO: remove pandas dependency


def keras_data_config(mappings_path, column, root, MAXLEN=2000, NUM_WORDS=20000, category_threshold=100):
    """
    data handler utility function

    Parameters: 

    mappings_path: str
        path to document_ID/file_name csv (legal/credits csv)

    column: str
        "Doc_Type" or "Doc_Subtype", label-encoding parameter

    root: str
        root path directory

    MAXLEN: int
        maximum length of the sequence to be used by keras tokenizer

    NUM_WORDS: int
        maximum number of words to be used by keras tokenizer

    category_threshold: int
        minimum number of samples necessary for a label

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

    df = load_data(mappings_path, column, root,
                   category_threshold=category_threshold)
    texts = [t for t in df.text]
    labels = [l for l in df.label]
    num_class = len(set(x for x in labels))
    x_train, x_test, y_train, y_test = train_test_split_data(texts, labels)
    X_train, X_val, Y_train, Y_val, tokenizer, word_index = keras_text_tokenizer(x_train, x_test,
                                                                                 y_train, y_test,
                                                                                 num_class,
                                                                                 NUM_WORDS=NUM_WORDS,
                                                                                 MAXLEN=MAXLEN)
    return X_train, X_val, Y_train, Y_val, tokenizer, word_index


def keras_model_config(pretrained_embeddings_path, word_index,
                       num_class, model_type="vanilla", embedding_type="glove", EMBEDDING_DIM=300,
                       MAX_WORDS=20000, MAX_SEQUENCE_LENGTH=1000, pretrained_weights=True, activation='relu', loss='categorical_crossentropy',
                       optimizer='adam', metrics=['acc'], batch_shape=None, sparse=False, lr=0.001,
                       beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, drop=0.5, num_filters=100, filter_sizes=[3, 4, 5]):
    """keras model configuration utility function

    Parameters: 

    pretrained_embeddings_path: str
        path to pretrained glove embeddings.

    model_type: str
        denotes which classification model to use,
         choose between "kim" and "vanilla"

    embedding_type: str
        denotes which embeddings model to use,
         choose between "glove" and "word2vec"

    word_index: dict
        an index of words from the dataset,
         returned by the keras_tokenizer.

    MAX_WORDS: int
        max number of words to be included in the processing.

    EMBEDDING_DIM: int
        dimension of the embedding layer.

    MAX_SEQUENCE_LENGTH: int
        maximum length of the sequence to be considered.

    pretrained_weights: bool
        if `true` returns the layer with pretrainned embeddings, 
        if `false` returns a layer without pretrained embeddings.

    embedding_layer: keras object
        keras embedding layer object, returned from embedding_utils module

    num_class: int
        number of classification categories

    lr: float
        learning rate

    beta_1: float
        0 < beta < 1. Generally close to 1.

    beta_2: float 
        0 < beta < 1. Generally close to 1.

    epsilon: float 
        epsilon >= 0Fuzz factor. If None, defaults to K.epsilon()

    decay: float
        decay >= 0. Learning rate decay over each update.

    drop: float
        dropout size

    num_filters: int
        number of filters to be used while reshaping the data

    filter_sizes: list
        (by default uses the original architecture values,
         best left unchanged) size of a filter

    loss: str
        type of loss function to be used

    metrics: list
        types of evaluation metrics to be considered,
         for the complete list of accepted values check keras documentation.

    optimizer: str
        type of optimizer function to be used.
         for a complete list please check keras documentation.

    batch_shape: optional/int
        bastch shape of the data to be used

    sparse: optional/bool
        specifies if the data is sparsely shaped or not.


    """

    if embedding_type == "glove":
        emb_lay = glove_embedding_layer(pretrained_embeddings_path,
                                        word_index,
                                        EMBEDDING_DIM=EMBEDDING_DIM,
                                        MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                                        MAX_WORDS=MAX_WORDS,
                                        pretrained_weights=pretrained_weights)
    if embedding_type == "word2vec":
        emb_lay = word2vec_embedding_layer(pretrained_embeddings_path, word_index,
                                           EMBEDDING_DIM=EMBEDDING_DIM,
                                           NUM_WORDS=MAX_WORDS,
                                           pretrained_weights=pretrained_weights)
    if model_type == "vanilla":
        model = vanilla_cnn(max_seq_len=MAX_SEQUENCE_LENGTH,
                            n_class=num_class,
                            embedding_layer=emb_lay,
                            activation=activation,
                            loss=loss,
                            optimizer=optimizer,
                            metrics=metrics,
                            batch_shape=batch_shape,
                            sparse=sparse)

    if model_type == "kim":
        model = kim_cnn(sequence_length=MAX_SEQUENCE_LENGTH,
                        EMBEDDING_DIM=EMBEDDING_DIM,
                        embedding_layer=emb_lay,
                        n_class=num_class,
                        lr=lr,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        decay=decay,
                        drop=drop,
                        num_filters=num_filters,
                        filter_sizes=filter_sizes,
                        loss=loss,
                        metrics=metrics)

    return model

# TODO: romove pandas dependency


def sklearn_data_config(mappings_path, column, root, category_threshold=100,
                        w2v_path=None, processing_type="tfidf", split_size=0.2,
                        encoding='utf-8', decode_error='strict', preprocessor=None, tokenizer=None, analyzer='word', stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None,
                        binary=False, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False, limit=200000, pos_filter=['ADJ', 'NOUN']):
    """
    utility function to create the vector representation 
    of the input data files using tfidf/word2vec sklearn model. 
    this function needs to be used when you want to train a 
    sklearn based classifier.

    note: this is a util function that wraps around 
    `vectorizer` & `train_test_split_data` functions. 

    Parameters:

    mappings_path: str
        path to document_ID/file_name csv (legal/credits csv)

    column: str
        "Doc_Type" or "Doc_Subtype", label-encoding parameter

    root: str
        root path directory

    category_threshold: int
        minimum number of samples necessary for a label

    w2v_path: str/None
        path to pretrained word embeddings

    processing_type: str
        type of text processing, choose inbetween `tfidf` and word2vec`
        note: if you choose word2vec, you cannot use it with 
        any bayesian models(GNB/MNB)

    split_size: float
        test data split size

    encoding: str
        type of encoding

    decode_error: str
        how to handle the decoding error, choose one of these: 
        'strict', 'ignore', 'replace'

    analyzer : string, {'word', 'char'} or callable
        Whether the feature should be made of word or character n-grams. 
        If a callable is passed it is used to extract the sequence of
         features out of the raw, unprocessed input.  

    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while 
        preserving the tokenizing and n-grams generation steps.  

    tokenizer : callable or None (default)
        Override the string tokenization step while preserving 
        the preprocessing and n-grams generation steps. 
        Only applies if `analyzer == 'word'`.  

    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of
         n-values for different n-grams to be extracted. 
         All values of n such that min_n  = n  = max_n will be used.  

    stop_words : string {'english'}, list, or None (default)
        If a string, it is passed to _check_stop_list and the 
        appropriate stop list is returned. 'english' is 
        currently the only supported string value. 
        If a list, that list is assumed to contain stop words, 
        all of which will be removed from the resulting tokens.
        Only applies if `analyzer == 'word'`.  

        If None, no stop words will be used. max_df 
        can be set to a value in the range [0.7, 1.0) 
        to automatically detect and filter stop words
        based on intra corpus document frequency of terms.  

    token_pattern : string
        Regular expression denoting what constitutes a "token", 
        only used if `analyzer == 'word'`. The default regexp 
        selects tokens of 2 or more alphanumeric characters 
        (punctuation is completely ignored and always 
        treated as a token separator).  

    max_df : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a 
        document frequency strictly higher than the given 
        threshold (corpus-specific stop words).If float, 
        the parameter represents a proportion of documents, 
        integer absolute counts.This parameter is ignored 
        if vocabulary is not None.  

    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have 
        a document frequency strictly lower than the given 
        threshold. This value is also called cut-off in the literature. 
        If float, the parameter represents a proportion of documents, 
        integer absolute counts. 
        This parameter is ignored if vocabulary is not None. 

    max_features : int or None, default=None 
        If not None, build a vocabulary that only consider 
        the top max_features ordered by term frequency 
        across the corpus. 
        This parameter is ignored if vocabulary is not None. 

    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys
         are terms and values are indices in the feature matrix, 
         or an iterable over terms. If not given,
          a vocabulary is determined from the input documents.  

    binary : boolean, default=False
        If True, all non-zero term counts are set to 1. 
        This does not mean outputs will have only 0/1 values,
        only that the tf term in tf-idf is binary.
        (Set idf and normalization to False to get 0/1 outputs.)

    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().  

    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.  

    use_idf : boolean, default=True
        Enable inverse-document-frequency reweighting.  

    smooth_idf : boolean, default=True
        Smooth idf weights by adding one to document frequencies,
         as if an extra document was seen containing every term
          in the collection exactly once. Prevents zero divisions.  

    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    limit: int
        max number of words

    pos_filter: list
        list of pos tags to keep in the output embeddings

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

    data = load_data(mappings_path, column, root,
                     category_threshold=category_threshold)

    if processing_type == "tfidf":
        train_data, test_data, \
            train_labels, test_labels, \
            fitted_model = vectorized_data(data.text, data.label, test_size=split_size, encoding=encoding, decode_error=decode_error, preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer, stop_words=stop_words,
                                           token_pattern=token_pattern, ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features, vocabulary=vocabulary, binary=binary, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)

        return train_data, test_data, train_labels, test_labels, fitted_model

    if processing_type == "word2vec":

        x_data, y_data = document_pos_tagger(data)
        x_tr, x_te, y_tr, y_te = train_test_split_data(
            x_data, y_data, test_size=split_size)
        train_data = word2vec(x_tr, w2v_path, limit=limit,
                              pos_filter=pos_filter)
        train_labels = np.array(y_tr)
        test_data = word2vec(x_te, w2v_path, limit=limit,
                             pos_filter=pos_filter)
        test_labels = np.array(y_te)
        fitted_model = "please choose tfidf to get a fitted model"
        return train_data, test_data, train_labels, test_labels, fitted_model


def sklearn_model_config(name, config=None):
    """sklearn based model configuration utility

    Parameters:

    name: str
        classifier name, should be one of 
        {'RF', 'MNB', 'GNB', 'SVC', 'MLP', 'AdaBoost', 'QDA', 'GPC', 'ET'}

    config: dict/None
        classifier hyperparameters dictionary

    note: for full hyperparams list check the corresponding sklearn documentation.

    Returns: 

    sklearn model object

    """
    model = classifier(name, params=config).model()
    return model
