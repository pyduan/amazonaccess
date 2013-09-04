"""combine.py

This is an ad-hoc script we used to find how to merge our submissions.
For this to work, the prediction vectors must be placed in the internal/
folder.

Author: Paul Duan <email@paulduan.com>
"""

import numpy as np
import math
from sklearn import linear_model, cross_validation, preprocessing

from ..helpers.data import load_data
from ..helpers.ml import compute_auc, AUCRegressor


def inverse_transform(X):
    def clamp(x):
        return min(max(x, .00000001), .99999999)
    return np.vectorize(lambda x: -math.log((1 - clamp(x))/clamp(x)))(X)


def print_param(obj, params, prefix=''):
    for param in params:
        if hasattr(obj, param):
            paramvalue = getattr(obj, param)
            if "coef" in param:
                paramvalue /= np.sum(paramvalue)
            print prefix + param + ": " + str(paramvalue)


mean_prediction = 0.0
y = load_data('train.csv')[0]
y = y[range(len(y) - 7770, len(y))]

files = ["log75", "ens", "paul"]
totransform = []

preds = []
for filename in files:
    with open("%s.csv" % filename) as f:
        pred = np.loadtxt(f, delimiter=',', usecols=[1], skiprows=1)
        if filename in totransform:
            pred = inverse_transform(pred)
        preds.append(pred)
X = np.array(preds).T

standardizer = preprocessing.StandardScaler()
X = standardizer.fit_transform(X)

print "============================================================"
print '\t\t'.join(files)
aucs = []
for filename in files:
    with open("%s.csv" % filename) as f:
        pred = np.loadtxt(f, delimiter=',', usecols=[1], skiprows=1)
        aucs.append("%.3f" % (compute_auc(y, pred) * 100))
print '\t\t'.join(aucs)
print "------------------------------------------------------------"

combiners = [
    linear_model.LinearRegression(),
    linear_model.Ridge(20),
    AUCRegressor(),
]

for combiner in combiners:
    mean_coefs = 0.0
    mean_auc = 0.0
    N = 10

    print "\n%s:" % combiner.__class__.__name__
    if hasattr(combiner, 'predict_proba'):
        combiner.predict = lambda X: combiner.predict_proba(X)[:, 1]

    combiner.fit(X, y)
    print_param(combiner, ["alpha_", "coef_"], "(post) ")
    print "Train AUC: %.3f" % (compute_auc(y, combiner.predict(X)) * 100)

    if isinstance(combiner, AUCRegressor):
        continue

    kfold = cross_validation.KFold(len(y), 3, shuffle=True)
    for train, test in kfold:
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]

        combiner.fit(X_train, y_train)
        prediction = combiner.predict(X_test)
        mean_auc += compute_auc(y_test, prediction)/len(kfold)

        if len(combiner.coef_) == 1:
            mean_coefs += combiner.coef_[0]/len(files)
        else:
            mean_coefs += combiner.coef_/len(files)

    print "Mean AUC: %.3f" % (mean_auc * 100)

print "\n------------------------------------------------------------"
