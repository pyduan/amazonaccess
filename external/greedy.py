""" Greedy feature selection
This file is a slightly modified version of Miroslaw's code.
It generates a dataset containing all 3rd order combinations
of the original columns, then performs greedy feature selection.

Original author: Miroslaw Horbal <miroslaw@gmail.com>
Permission was granted by Miroslaw to publish this snippet as part of
our code.
"""

from sklearn import metrics, cross_validation, linear_model
from scipy import sparse
from itertools import combinations
from helpers import data

import numpy as np
import pandas as pd

SEED = 333


def group_data(data, degree=3, hash=hash):
    new_data = []
    m, n = data.shape
    for indices in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:, indices]])
    return np.array(new_data).T


def OneHotEncoder(data, keymap=None):
    """
    OneHotEncoder takes data matrix with categorical columns and
    converts it to a sparse binary matrix.

    Returns sparse binary matrix and keymap mapping categories to indicies.
    If a keymap is supplied on input it will be used instead of creating one
    and any categories appearing in the data that are not in the keymap are
    ignored
    """
    if keymap is None:
        keymap = []
        for col in data.T:
            uniques = set(list(col))
            keymap.append(dict((key, i) for i, key in enumerate(uniques)))
    total_pts = data.shape[0]
    outdat = []
    for i, col in enumerate(data.T):
        km = keymap[i]
        num_labels = len(km)
        spmat = sparse.lil_matrix((total_pts, num_labels))
        for j, val in enumerate(col):
            if val in km:
                spmat[j, km[val]] = 1
        outdat.append(spmat)
    outdat = sparse.hstack(outdat).tocsr()
    return outdat, keymap


def cv_loop(X, y, model, N):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y, test_size=.20,
            random_state=i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:, 1]
        auc = metrics.auc_score(y_cv, preds)
        print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
        mean_auc += auc
    return mean_auc/N


def create_features(train='data/train.csv', test='data/test.csv'):
    print "Reading dataset..."
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)
    all_data = np.vstack((train_data.ix[:, 1:-1], test_data.ix[:, 1:-1]))

    num_train = np.shape(train_data)[0]

    # Transform data
    print "Transforming data..."
    dp = group_data(all_data, degree=2)
    dt = group_data(all_data, degree=3)

    y = np.array(train_data.ACTION)
    X = all_data[:num_train]
    X_2 = dp[:num_train]
    X_3 = dt[:num_train]

    X_test = all_data[num_train:]
    X_test_2 = dp[num_train:]
    X_test_3 = dt[num_train:]

    X_train_all = np.hstack((X, X_2, X_3))
    X_test_all = np.hstack((X_test, X_test_2, X_test_3))
    num_features = X_train_all.shape[1]

    model = linear_model.LogisticRegression()

    # Xts holds one hot encodings for each individual feature in memory
    # speeding up feature selection
    Xts = [OneHotEncoder(X_train_all[:, [i]])[0] for i in range(num_features)]

    print "Performing greedy feature selection..."
    score_hist = []
    N = 10
    good_features_list = [
        [0, 8, 9, 10, 19, 34, 36, 37, 38, 41, 42, 43, 47, 53, 55,
         60, 61, 63, 64, 67, 69, 71, 75, 81, 82, 85],
        [0, 1, 7, 8, 9, 10, 36, 37, 38, 41, 42, 43, 47, 51, 53,
         56, 60, 61, 63, 64, 66, 67, 69, 71, 75, 79, 85, 91],
        [0, 7, 9, 24, 36, 37, 41, 42, 47, 53, 61, 63, 64, 67, 69, 71, 75, 85],
        [0, 7, 9, 20, 36, 37, 38, 41, 42, 45, 47,
         53, 60, 63, 64, 67, 69, 71, 81, 85, 86]
    ]

    # Greedy feature selection loop
    if not good_features_list:
        good_features = set([])
        while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
            scores = []
            for f in range(len(Xts)):
                if f not in good_features:
                    feats = list(good_features) + [f]
                    Xt = sparse.hstack([Xts[j] for j in feats]).tocsr()
                    score = cv_loop(Xt, y, model, N)
                    scores.append((score, f))
                    print "Feature: %i Mean AUC: %f" % (f, score)
            good_features.add(sorted(scores)[-1][1])
            score_hist.append(sorted(scores)[-1])
            print "Current features: %s" % sorted(list(good_features))

        # Remove last added feature from good_features
        good_features.remove(score_hist[-1][1])
        good_features = sorted(list(good_features))

    for i, good_features in enumerate(good_features_list):
        suffix = str(i + 1) if i else ''
        Xt = np.vstack((X_train_all[:, good_features],
                        X_test_all[:, good_features]))
        X_train = Xt[:num_train]
        X_test = Xt[num_train:]
        data.save_dataset("greedy%s" % suffix, X_train, X_test)
