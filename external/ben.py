""" Amazon Access Challenge Starter Code

This was built using the code of Paul Duan <email@paulduan.com> as a starting
point (thanks to Paul).

It builds ensemble models using the original dataset and a handful of
extracted features.

Author: Benjami Solecki <bensolucky@gmail.com>
"""

from __future__ import division

import numpy as np
import pandas as pd
from helpers.data import save_dataset


def create_features():
    print "loading data"
    X = pd.read_csv('data/train.csv')
    X = X.drop(['ROLE_CODE'], axis=1)
    X = X.drop(['ACTION'], axis=1)

    X_test = pd.read_csv('data/test.csv', index_col=0)
    X_test = X_test.drop(['ROLE_CODE'], axis=1)
    X_test['ACTION'] = 0
    X_test = X_test.drop(['ACTION'], axis=1)

    X_all = pd.concat([X_test, X], ignore_index=True)
    # I want to combine role_title as a subset of role_familia and
    X_all['ROLE_TITLE'] = X_all['ROLE_TITLE'] + (1000 * X_all['ROLE_FAMILY'])
    X_all['ROLE_ROLLUPS'] = X_all['ROLE_ROLLUP_1'] + (
        10000 * X_all['ROLE_ROLLUP_2'])
    X_all = X_all.drop(['ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_FAMILY'],
                       axis=1)

    # Count/freq
    for col in X_all.columns:
        X_all['cnt'+col] = 0
        groups = X_all.groupby([col])
        for name, group in groups:
            count = group[col].count()
            X_all['cnt'+col].ix[group.index] = count
        X_all['cnt'+col] = X_all['cnt'+col].apply(np.log)

    # Percent of dept that is this resource
    # And Counts of dept/resource occurancesa (tested, not used)
    for col in X_all.columns[1:6]:
        X_all['Duse'+col] = 0.0
        groups = X_all.groupby([col])
        for name, group in groups:
            grps = group.groupby(['RESOURCE'])
            for rsrc, grp in grps:
                X_all['Duse'+col].ix[grp.index] = \
                    float(len(grp.index)) / float(len(group.index))

    # Number of resources that a manager manages
    for col in X_all.columns[0:1]:
    #for col in X_all.columns[0:6]:
        if col == 'MGR_ID':
            continue
        X_all['Mdeps'+col] = 0
        groups = X_all.groupby(['MGR_ID'])
        for name, group in groups:
            X_all['Mdeps'+col].ix[group.index] = len(group[col].unique())

    X_all = X_all.drop(X_all.columns[0:6], axis=1)

    # Now X is the train, X_test is test and X_all is both together
    X = X_all[:][X_all.index >= len(X_test.index)]
    X_test = X_all[:][X_all.index < len(X_test.index)]
    # X is the train set alone, X_all is all features
    X = X.as_matrix()
    X_test = X_test.as_matrix()

    save_dataset('bsfeats', X, X_test)
