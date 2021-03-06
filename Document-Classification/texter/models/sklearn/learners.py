from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import cross_validation

import warnings
warnings.simplefilter("ignore")


options = {'RF': RandomForestClassifier,
           'MNB': MultinomialNB,
           'GNB': GaussianNB,
           'SVC': SVC,
           'MLP': MLPClassifier,
           'AdaBoost': AdaBoostClassifier,
           'QDA': QuadraticDiscriminantAnalysis,
           'GPC': GaussianProcessClassifier,
           'ET': ExtraTreesClassifier}


class classifier():
    """
    texter base classifier class, returns a sklearn.model object,
    configure it with the hyperparams of the selected classifier

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

    def __init__(self, name, params=None):
        self.name = name
        self.params = params
        clf = options.get(name)()
        if params:
            clf = options.get(name)(**params)
        print(
            f"classification model configured to use {clf.__class__.__name__} algorithm.")
        self._clf = clf

    def model(self):
        return self._clf
