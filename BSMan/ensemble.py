""" Amazon Access Challenge Starter Code

This was built using the code of Paul Duan <email@paulduan.com> as a starting
point (thanks to Paul).

It builds ensemble models using the original dataset and a handful of 
extracted features.

Author: Benjamin Solecki <bensolecki@gmail.com>
"""

from __future__ import division

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn import (metrics, cross_validation, linear_model, preprocessing)

SEED = 42  # always use a seed for randomized procedures

def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))


"""
Fit models and make predictions.
We'll use one-hot encoding to transform our categorical features
into binary features.
y and X will be numpy array objects.
"""
# === load data in memory === #
print "loading data"
X = pd.read_csv('data/train.csv')
X = X.drop(['ROLE_CODE'], axis=1)
y = X['ACTION']
X = X.drop(['ACTION'], axis=1)
X_test = pd.read_csv('data/test.csv', index_col=0)
X_test = X_test.drop(['ROLE_CODE'], axis=1)
X_test['ACTION'] = 0
y_test = X_test['ACTION']
X_test = X_test.drop(['ACTION'], axis=1)

modelRF =RandomForestClassifier(n_estimators=1999, max_features='sqrt', max_depth=None, min_samples_split=9, compute_importances=True, random_state=SEED)#8803
modelXT =ExtraTreesClassifier(n_estimators=1999, max_features='sqrt', max_depth=None, min_samples_split=8, compute_importances=True, random_state=SEED) #8903
modelGB =GradientBoostingClassifier(n_estimators=50, learning_rate=0.20, max_depth=20, min_samples_split=9, random_state=SEED)  #8749
# 599: 20/90/08
#1999: 24/95/06

X_all = pd.concat([X_test,X], ignore_index=True)

# I want to combine role_title as a subset of role_familia and see if same results
X_all['ROLE_TITLE'] = X_all['ROLE_TITLE'] + (1000 * X_all['ROLE_FAMILY'])
X_all['ROLE_ROLLUPS'] = X_all['ROLE_ROLLUP_1'] + (10000 * X_all['ROLE_ROLLUP_2'])
X_all = X_all.drop(['ROLE_ROLLUP_1','ROLE_ROLLUP_2','ROLE_FAMILY'], axis=1)

# Count/freq
print "Counts"
for col in X_all.columns:
    X_all['cnt'+col] = 0
    groups = X_all.groupby([col])
    for name, group in groups:
        count = group[col].count()
        X_all['cnt'+col].ix[group.index] = count 
    X_all['cnt'+col] = X_all['cnt'+col].apply(np.log) # could check if this is neccesary, I think probably not

# Percent of dept that is this resource
for col in X_all.columns[1:6]:
    X_all['Duse'+col] = 0.0
    groups = X_all.groupby([col])
    for name, group in groups:
        grps = group.groupby(['RESOURCE'])
        for rsrc, grp in grps:
            X_all['Duse'+col].ix[grp.index] = float(len(grp.index)) / float(len(group.index) )

# Number of resources that a manager manages
for col in X_all.columns[0:1]:
    if col == 'MGR_ID':
        continue
    print col
    X_all['Mdeps'+col] = 0
    groups = X_all.groupby(['MGR_ID'])
    for name, group in groups:
        X_all['Mdeps'+col].ix[group.index] = len(group[col].unique()) 


X = X_all[:][X_all.index>=len(X_test.index)]
X_test = X_all[:][X_all.index<len(X_test.index)]

# === Combine Models === #
# Do a linear combination using a cross_validated data split
X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=0.5, random_state=SEED)

modelRF.fit(X_cv, y_cv) 
modelXT.fit(X_cv, y_cv) 
modelGB.fit(X_cv, y_cv) 
predsRF = modelRF.predict_proba(X_train)[:, 1]
predsXT = modelXT.predict_proba(X_train)[:, 1]
predsGB = modelGB.predict_proba(X_train)[:, 1]
preds = np.hstack((predsRF, predsXT, predsGB)).reshape(3,len(predsGB)).transpose()
preds[preds>0.9999999]=0.9999999
preds[preds<0.0000001]=0.0000001
preds = -np.log((1-preds)/preds)
modelEN1 = linear_model.LogisticRegression()
modelEN1.fit(preds, y_train)
print modelEN1.coef_

modelRF.fit(X_train, y_train) 
modelXT.fit(X_train, y_train) 
modelGB.fit(X_train, y_train) 
predsRF = modelRF.predict_proba(X_cv)[:, 1]
predsXT = modelXT.predict_proba(X_cv)[:, 1]
predsGB = modelGB.predict_proba(X_cv)[:, 1]
preds = np.hstack((predsRF, predsXT, predsGB)).reshape(3,len(predsGB)).transpose()
preds[preds>0.9999999]=0.9999999
preds[preds<0.0000001]=0.0000001
preds = -np.log((1-preds)/preds)
modelEN2 = linear_model.LogisticRegression()
modelEN2.fit(preds, y_cv)
print modelEN2.coef_

coefRF = modelEN1.coef_[0][0] + modelEN2.coef_[0][0]
coefXT = modelEN1.coef_[0][1] + modelEN2.coef_[0][1]
coefGB = modelEN1.coef_[0][2] + modelEN2.coef_[0][2]

# === Predictions === #
# When making predictions, retrain the model on the whole training set
modelRF.fit(X, y)
modelXT.fit(X, y)
modelGB.fit(X, y)

### Combine here
predsRF = modelRF.predict_proba(X_test)[:, 1]
predsXT = modelXT.predict_proba(X_test)[:, 1]
predsGB = modelGB.predict_proba(X_test)[:, 1]
predsRF[predsRF>0.9999999]=0.9999999
predsXT[predsXT>0.9999999]=0.9999999
predsGB[predsGB>0.9999999]=0.9999999
predsRF[predsRF<0.0000001]=0.0000001
predsXT[predsXT<0.0000001]=0.0000001
predsGB[predsGB<0.0000001]=0.0000001
predsRF = -np.log((1-predsRF)/predsRF)
predsXT = -np.log((1-predsXT)/predsXT)
predsGB = -np.log((1-predsGB)/predsGB)
preds = coefRF * predsRF + coefXT * predsXT + coefGB * predsGB

filename = raw_input("Enter name for submission file: ")
save_results(preds, "submissions/en" + filename + ".csv")
