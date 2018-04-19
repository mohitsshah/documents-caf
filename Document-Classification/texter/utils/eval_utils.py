from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import cross_validation

__all__ = ['classifier_report']


def classifier_report(model, x_test, y_test, y_pred):
    score = model.score(x_test, y_test)
    model_report = classification_report(y_test, y_pred)
    model_confusion_matrix = confusion_matrix(y_test, y_pred)
    return f"1. score: {score}\n2. classification model report:\n{model_report}\n3. confusion matrix:\n{model_confusion_matrix}"
