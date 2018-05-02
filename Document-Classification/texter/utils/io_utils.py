import os
import json
import re
import sys
import glob
import pathlib

import pandas as pd
import numpy as np

from keras.models import model_from_json
from sklearn.externals import joblib
from nltk.corpus import stopwords

from .externals import data_loader, return_df

__all__ = ['load_config', 'save_config', 'load_model',
           'save_model', 'load_text_model', 'save_text_model']

# TODO: remove pandas dependency


def load_data(mappings_path, column, root, category_threshold=100):
    """

    Dataset loader utility
    NOTE: Will be depricated in the next version

    Parameters: 

    mappings_path: str
        path to document_ID/file_name csv

    column: str
        "Doc_Type" or "Doc_Subtype", label-encoding parameter

    root: str
        root path directory

    Returns: pandas dataframe

    """
    texts, labels = data_loader(mappings_path, column, root)
    df = pd.DataFrame(dict(text=texts, label=labels))
    df.text = df.text.replace('', np.nan)
    df = df.dropna().reset_index()
    return return_df(df, column, category_threshold)


# """
# def load_data(text_directory_path, class_labels, remove_pattern="[^a-zA-Z]"):
#    """
#    loads the data as a dataframe containing the labels and text associated with it
#
#    Parameters:
#
#    text_directory_path: str
#        path to directory containing the text files
#
#    class_labels: list
#        list of labels
#
#    remove_pattern: str
#        regex pattern to remove unwanted chars from the text
#
#    Returns:
#        Dataframe
#
#    """
#    excludes = stopwords.words("english")
#    pattern = re.compile(remove_pattern)
#    texts = []
#    labels = []
#    textdir = text_directory_path
#    categories = class_labels
#    for i, category in enumerate(categories):
#        path = os.path.join(textdir, category)
#        files = os.listdir(path)
#        for f in files:
#            p = os.path.join(path, f)
#            content = open(p, 'r').read()
#            content = pattern.sub(" ", content)
#            words = content.split()
#            words = [w for w in words if len(w) > 1]
#            words = [w for w in words if w not in excludes]
#            if len(words) > 0:
#                lines = ' '.join(words)
#                texts.append(lines)
#                labels.append(i)
#    return pd.DataFrame(dict(text=texts, label=labels))
# """


def load_config(filepath):
    """
    utility function to load the hyperparams/data-config 
    from a saved json file

    Parameters: 

    filepath: str
        path to the json file containing the hyperparams/data-config.

    Returns: 

        returns a config dictionary
    """
    with open(filepath) as json_data:
        return json.load(json_data)


def save_config(filepath, config_dict):
    """
    utility function to save the hyperparams/data-config to a json file

    Parameters: 

    filepath: str
        path to save the  json file containing the hyperparams.(please provide the filename as well)

    config_dict: dict
        hyperparams dictionary

    Returns: 

        None

    """
    with open(filepath, 'w') as fp:
        json.dump(config_dict, fp)


def save_model(name, model, overwrite=True):
    """
    utility function to save the classifier object(and weights) to disk.
    note: keras model gets saved as a json file, with weights saved as an h5 file.
    and sklearn model gets saved as a pickle file.
    (do not pass file extension)

    Parameters: 

    name: str
        classifier name

    model: sklearn/keras clf object
        classifier object to be saved

    overwrite: Bool
        overwrite existing model, with same name or give the user a prompt to change the name.

    """
    try:
        model.save_weights(f"{name}_weights.h5", overwrite=overwrite)
        with open(f'{name}_architecture.json', 'w') as f:
            f.write(model.to_json())
    except Exception:
        joblib.dump(model, f'{name}.pkl')


def load_model(filename):
    """
    utility function to load the saved model(and weights).
    (do not pass file extension)

    Parameters: 

    filename: str
        path/file name

    Returns: 
        sklearn/keras model object

    """
    try:
        with open(f'{filename}_architecture.json', 'r') as f:
            model = model_from_json(f.read())
        model.load_weights(f'{filename}_weights.h5')
    except Exception:
        model = joblib.load(f'{filename}.pkl')

    return model


def save_text_model(name, text_model):
    """
    utility function to save text processing model
    note: (do not pass file extension)

    Parameters: 

    name: str
        filepath/filename (do not provide the file extension)

    text_model: sklearn object
        vectorizer object from sklearn or a string(if processing type="word2vec")

    """
    joblib.dump(text_model, f'{name}_tokenizer.pkl')


def load_text_model(name):
    """
    utility function to load the text processing model

    Parameters:

    name: str
        filepath/filename (do not provide the file extension)

    Returns: 

    sklearn vectorizer object    
    """
    return joblib.load(f'{name}_tokenizer.pkl')
