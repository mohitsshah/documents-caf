from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import cross_validation
from keras.callbacks import EarlyStopping

import warnings
warnings.simplefilter("ignore")

__all__ = ['classifier_report', 'add_callbacks']


def classifier_report(model, x_test, y_test, y_pred):
    """
    Returns the summary of a classifier's performance.

    Parameters:

    model: sklearn object
        sklearn fitted classifier

    x_test: numpy array
        test samples

    y_test: numpy array
        test labels

    y_pred: numpy array
        predicted labels/outcomes

    Returns: 

    classifier report (str)
    """
    score = model.score(x_test, y_test)
    model_report = classification_report(y_test, y_pred)
    model_confusion_matrix = confusion_matrix(y_test, y_pred)
    return f"1. score: {score}\n2. classification model report:\n{model_report}\n3. confusion matrix:\n{model_confusion_matrix}"


def add_callbacks(monitor='val_acc', min_delta=0.0001, patience=5,
                          verbose=1, mode='auto'):
    """
    utility function that returns a callback list, 
    that can be passed to a keras model definition

    Parameters: 

    monitor: str
        quantity to be monitored.  

    min_delta: float
        minimum change in the monitored quantity to qualify as 
        an improvement, i.e. an absolute change of less 
        than min_delta, will count as no improvement.  

    patience: int
        number of epochs with no improvement after which 
        training will be stopped.  

    verbose: int
        verbosity mode.

    mode: 
        one of {auto, min, max}. In `min` mode,training will stop 
        when the quantity monitored has stopped decreasing; 
        in `max` mode it will stop when the quantity monitored 
        has stopped increasing; in `auto` mode, 
        the direction is automatically inferred from 
        the name of the monitored quantity. 

    Returns: list 
        list of callbacks


    """
    earlystop = EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience,
                              verbose=verbose, mode=mode)
    callbacks_list = [earlystop]
    return callbacks_list
