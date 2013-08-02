"""utils.py

Some useful functions.
Author: Paul Duan <email@paulduan.com>
"""

from re import sub
from sklearn.metrics import roc_curve, auc


def stringify(model, feature_set):
    """Given a model and a feature set, return a short string that will serve
    as identifier for this combination.
    Ex: (LogisticRegression(), "basic_s") -> "LR:basic_s"
    """
    return "%s:%s" % (sub("[a-z]", '', model.__class__.__name__), feature_set)


def compute_auc(y, y_pred):
    fpr, tpr, _ = roc_curve(y, y_pred)
    return auc(fpr, tpr)
