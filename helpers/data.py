"""ml.py

Useful I/O functions.

Author: Paul Duan <email@paulduan.com>
"""

import logging
import numpy as np
from scipy import sparse
import cPickle as pickle

logger = logging.getLogger(__name__)


def load_data(filename, return_labels=True):
    """Load data from CSV files and return them in numpy format."""
    logging.debug("loading data from %s", filename)
    data = np.loadtxt(open("data/" + filename), delimiter=',',
                      usecols=range(1, 10), skiprows=1, dtype=int)
    if return_labels:
        labels = np.loadtxt(open("data/" + filename), delimiter=',',
                            usecols=[0], skiprows=1)
        return labels, data
    else:
        labels = np.zeros(data.shape[0])
        return data


def load_from_cache(filename, use_cache=True):
    """Attempt to load data from cache."""
    data = None
    read_mode = 'rb' if '.pkl' in filename else 'r'
    if use_cache:
        try:
            with open("cache/%s" % filename, read_mode) as f:
                data = pickle.load(f)
        except IOError:
            pass

    return data


def save_results(predictions, filename):
    """Save results in CSV format."""
    logging.info("saving data to file %s", filename)
    with open("submissions/%s" % filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))


def save_dataset(filename, X, X_test, features=None, features_test=None):
    """Save the training and test sets augmented with the given features."""
    if features is not None:
        assert features.shape[1] == features_test.shape[1], "features mismatch"
        if sparse.issparse(X):
            features = sparse.lil_matrix(features)
            features_test = sparse.lil_matrix(features_test)
            X = sparse.hstack((X, features), 'csr')
            X_test = sparse.hstack((X_test, features_test), 'csr')
        else:
            X = np.hstack((X, features))
            X_test = np. hstack((X_test, features_test))

    logger.info("> saving %s to disk", filename)
    with open("cache/%s.pkl" % filename, 'wb') as f:
        pickle.dump((X, X_test), f, pickle.HIGHEST_PROTOCOL)


def get_dataset(feature_set='basic', train=None, cv=None):
    """
    Return the design matrices constructed with the specified feature set.
    If train is specified, split the training set according to train and
    cv (if cv is not given, subsample's complement will be used instead).
    If subsample is omitted, return both the full training and test sets.
    """
    try:
        with open("cache/%s.pkl" % feature_set, 'rb') as f:
            if train is not None:
                X, _ = pickle.load(f)
                if cv is None:
                    cv = [i for i in range(X.shape[0]) if i not in train]

                X_test = X[cv, :]
                X = X[train, :]
            else:
                X, X_test = pickle.load(f)
    except IOError:
        logging.warning("could not find feature set %s", feature_set)
        return False

    return X, X_test
