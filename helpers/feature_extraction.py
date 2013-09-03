"""feature_extraction.py

Create the requested datasets.

Author: Paul Duan <email@paulduan.com>
"""

from __future__ import division

import logging
import cPickle as pickle
import numpy as np
import math

from scipy import sparse
from sklearn import preprocessing

from external import greedy, ben
from data import save_dataset
from ml import get_dataset

logger = logging.getLogger(__name__)
subformatter = logging.Formatter("[%(asctime)s] %(levelname)s\t> %(message)s")

COLNAMES = ["resource", "manager", "role1", "role2", "department",
            "title", "family_desc", "family"]
SELECTED_COLUMNS = [0, 1, 4, 5, 6, 7]

EXTERNAL_DATASETS = {
    "greedy": greedy,
    "greedy2": greedy,
    "greedy3": greedy,
    "bsfeats": ben
}


def sparsify(X, X_test):
    """Return One-Hot encoded datasets."""
    enc = OneHotEncoder()
    enc.fit(np.vstack((X, X_test)))
    return enc.transform(X), enc.transform(X_test)


def create_datasets(X, X_test, y, datasets=[], use_cache=True):
    """
    Generate datasets as needed with different sets of features
    and save them to disk.
    The datasets are created by combining a base feature set (combinations of
    the original variables) with extracted feature sets, with some additional
    variants.

    The nomenclature is as follows:
    Base datasets:
        - basic: the original columns, minus role1, role2, and role_code
        - tuples: all order 2 combinations of the original columns
        - triples: all order 3 combinations of the original columns
        - greedy[1,2,3]: three different datasets obtained by performing
            greedy feature selection with different seeds on the triples
            dataset
        - effects: experimental. Created to try out a suggestion by Gxav
            after the competition

    Feature sets and variants:
    (denoted by the letters after the underscore in the base dataset name):
        - s: the base dataset has been sparsified using One-Hot encoding
        - c: the rare features have been consolidated into one category
        - f: extracted features have been appended, with a different set for
            linear models than for tree-based models
        - b: Benjamin's extracted features.
        - d: interactions for the extracted feature set have been added
        - l: the extracted features have been log transformed
    """
    if use_cache:
        # Check if all files exist. If not, generate the missing ones
        DATASETS = []
        for dataset in datasets:
            try:
                with open("cache/%s.pkl" % dataset, 'rb'):
                    pass
            except IOError:
                logger.warning("couldn't load dataset %s, will generate it",
                               dataset)
                DATASETS.append(dataset.split('_')[0])
    else:
        DATASETS = ["basic", "tuples", "triples",
                    "greedy", "greedy2", "greedy3"]

    # Datasets that require external code to be generated
    for dataset, module in EXTERNAL_DATASETS.iteritems():
        if not get_dataset(dataset):
            module.create_features()

    # Generate the missing datasets
    if len(DATASETS):
        bsfeats, bsfeats_test = get_dataset('bsfeats')

        basefeats, basefeats_test = create_features(X, X_test, 3)
        save_dataset("base_feats", basefeats, basefeats_test)

        lrfeats, lrfeats_test = pre_process(*create_features(X, X_test, 0))
        save_dataset("lrfeats", lrfeats, lrfeats_test)

        feats, feats_test = pre_process(*create_features(X, X_test, 1))
        save_dataset("features", feats, feats_test)

        meta, meta_test = pre_process(*create_features(X, X_test, 2),
                                      normalize=False)
        save_dataset("metafeatures", meta, meta_test)

        X = X[:, SELECTED_COLUMNS]
        X_test = X_test[:, SELECTED_COLUMNS]
        save_dataset("basic", X, X_test)

        Xt = create_tuples(X)
        Xt_test = create_tuples(X_test)
        save_dataset("tuples", Xt, Xt_test)

        Xtr = create_tuples(X)
        Xtr_test = create_tuples(X_test)
        save_dataset("triples", Xtr, Xtr_test)

        Xe, Xe_test = create_effects(X, X_test, y)
        save_dataset("effects", Xe, Xe_test)

        feats_d, feats_d_test = pre_process(basefeats, basefeats_test,
                                            create_divs=True)
        bsfeats_d, bsfeats_d_test = pre_process(bsfeats, bsfeats_test,
                                                create_divs=True)
        feats_l, feats_l_test = pre_process(basefeats, basefeats_test,
                                            log_transform=True)
        lrfeats_l, lrfeats_l_test = pre_process(lrfeats, lrfeats_test,
                                                log_transform=True)
        bsfeats_l, bsfeats_l_test = pre_process(bsfeats, bsfeats_test,
                                                log_transform=True)

        for ds in DATASETS:
            Xg, Xg_test = get_dataset(ds)
            save_dataset(ds + '_b', Xg, Xg_test, bsfeats, bsfeats_test)
            save_dataset(ds + '_f', Xg, Xg_test, feats, feats_test)
            save_dataset(ds + '_fd', Xg, Xg_test, feats_d, feats_d_test)
            save_dataset(ds + '_bd', Xg, Xg_test, bsfeats_d, bsfeats_d_test)
            Xs, Xs_test = sparsify(Xg, Xg_test)
            save_dataset(ds + '_sf', Xs, Xs_test, lrfeats, lrfeats_test)
            save_dataset(ds + '_sfl', Xs, Xs_test, lrfeats_l, lrfeats_l_test)
            save_dataset(ds + '_sfd', Xs, Xs_test, feats_d, feats_d_test)
            save_dataset(ds + '_sb', Xs, Xs_test, bsfeats, bsfeats_test)
            save_dataset(ds + '_sbl', Xs, Xs_test, bsfeats_l, bsfeats_l_test)
            save_dataset(ds + '_sbd', Xs, Xs_test, bsfeats_d, bsfeats_d_test)

            if issubclass(Xg.dtype.type, np.integer):
                consolidate(Xg, Xg_test)
                save_dataset(ds + '_c', Xg, Xg_test)
                save_dataset(ds + '_cf', Xg, Xg_test, feats, feats_test)
                save_dataset(ds + '_cb', Xg, Xg_test, bsfeats, bsfeats_test)
                Xs, Xs_test = sparsify(Xg, Xg_test)
                save_dataset(ds + '_sc', Xs, Xs_test)
                save_dataset(ds + '_scf', Xs, Xs_test, feats, feats_test)
                save_dataset(ds + '_scfl', Xs, Xs_test, feats_l, feats_l_test)
                save_dataset(ds + '_scb', Xs, Xs_test, bsfeats, bsfeats_test)
                save_dataset(ds + '_scbl', Xs, Xs_test,
                             bsfeats_l, bsfeats_l_test)


def create_effects(X_train, X_test, y):
    """
    Create a dataset where the features are the effects of a
    logistic regression trained on sparsified data.
    This has been added post-deadline after talking with Gxav.
    """
    from sklearn import linear_model, cross_validation
    from itertools import izip
    Xe_train = np.zeros(X_train.shape)
    Xe_test = np.zeros(X_test.shape)
    n_cols = Xe_train.shape[1]

    model = linear_model.LogisticRegression(C=2)
    X_train, X_test = sparsify(X_train, X_test)

    kfold = cross_validation.KFold(len(y), 5)
    for train, cv in kfold:
        model.fit(X_train[train], y[train])
        colindices = X_test.nonzero()[1]
        for i, k in izip(cv, range(len(cv))):
            for j in range(n_cols):
                z = colindices[n_cols*k + j]
                Xe_train[i, j] = model.coef_[0, z]

    model.fit(X_train, y)
    colindices = X_test.nonzero()[1]
    for i in range(Xe_test.shape[0]):
        for j in range(n_cols):
            z = colindices[n_cols*i + j]
            Xe_test[i, j] = model.coef_[0, z]

    return Xe_train, Xe_test


def create_features(X_train, X_test, feature_set=0):
    """
    Extract features from the training and test set.
    Each feature set is defined as a list of lambda functions.
    """
    logger.info("performing feature extraction (feature_set=%d)", feature_set)
    features_train = []
    features_test = []
    dictionaries = get_pivottable(X_train, X_test)
    dictionaries_train = get_pivottable(X_train, X_test, use='train')
    dictionaries_test = get_pivottable(X_test, X_test, use='test')

    # 0: resource, 1: manager, 2: role1, 3: role2, 4: department,
    # 5: title, 6: family_desc, 7: family
    feature_lists = [
        [  # 0: LR features
            lambda x, row, j:
            x[COLNAMES[0]].get(row[0], 0) if j > 0 and j < 7 else 0,
            lambda x, row, j:
            x[COLNAMES[1]].get(row[1], 0) if j > 1 and j < 7 else 0,
            lambda x, row, j:
            x[COLNAMES[2]].get(row[2], 0) if j > 2 and j < 7 else 0,
            lambda x, row, j:
            x[COLNAMES[3]].get(row[3], 0) if j > 3 and j < 7 else 0,
            lambda x, row, j:
            x[COLNAMES[4]].get(row[4], 0) if j > 4 and j < 7 else 0,
            lambda x, row, j:
            x[COLNAMES[5]].get(row[5], 0) if j > 5 and j < 7 else 0,
            lambda x, row, j:
            x[COLNAMES[6]].get(row[6], 0) if j > 6 and j < 7 else 0,
            lambda x, row, j:
            x[COLNAMES[7]].get(row[7], 0) if j > 7 and j < 7 else 0,

            lambda x, row, j:
            x[COLNAMES[0]].get(row[0], 0)**2 if j in range(7) else 0,
            lambda x, row, j:
            x[COLNAMES[j]].get(row[0], 0)/x['total']
            if j > 0 and j < 7 else 0,

            lambda x, row, j:
            x[COLNAMES[j]].get(row[j], 0)/len(x[COLNAMES[j]].values()),

            lambda x, row, j:
            x[COLNAMES[j]].get(row[j], 0) / dictionaries[j]['total'],

            lambda x, row, j:
            math.log(x[COLNAMES[0]].get(row[0], 0)) if j in range(5) else 0,

            lambda x, row, j:
            int(row[j] not in dictionaries_train[j]),

            lambda x, row, j:
            int(row[j] not in dictionaries_test[j]),
        ],

        [  # 1: Tree features
            lambda x, row, j:
            x[COLNAMES[0]].get(row[0], 0),
            lambda x, row, j:
            x[COLNAMES[1]].get(row[1], 0),
            lambda x, row, j:
            x[COLNAMES[2]].get(row[2], 0),
            lambda x, row, j:
            x[COLNAMES[3]].get(row[3], 0),
            lambda x, row, j:
            x[COLNAMES[4]].get(row[4], 0),
            lambda x, row, j:
            x[COLNAMES[5]].get(row[5], 0),
            lambda x, row, j:
            x[COLNAMES[6]].get(row[6], 0),
            lambda x, row, j:
            x[COLNAMES[7]].get(row[7], 0),

            lambda x, row, j:
            x[COLNAMES[j]].get(row[0], 0)/x['total'] if j > 0 else 0,
        ],

        [  # 2: Metafeatures
            lambda x, row, j:
            dictionaries_train[j].get(row[j], {}).get('total', 0),
            lambda x, row, j:
            dictionaries_train[j].get(row[j], {}).get('total', 0) == 0,
        ],

        [  # 3: Base features
            lambda x, row, j:
            x['total'] if j == 0 else 0,

            lambda x, row, j:
            x[COLNAMES[0]].get(row[0], 0) if j > 0 else 0,
            lambda x, row, j:
            x[COLNAMES[1]].get(row[1], 0) if j > 1 else 0,
            lambda x, row, j:
            x[COLNAMES[2]].get(row[2], 0) if j > 2 else 0,
            lambda x, row, j:
            x[COLNAMES[3]].get(row[3], 0) if j > 3 else 0,
            lambda x, row, j:
            x[COLNAMES[4]].get(row[4], 0) if j > 4 else 0,
            lambda x, row, j:
            x[COLNAMES[5]].get(row[5], 0) if j > 5 else 0,
            lambda x, row, j:
            x[COLNAMES[6]].get(row[6], 0) if j > 6 else 0,
            lambda x, row, j:
            x[COLNAMES[7]].get(row[7], 0) if j > 7 else 0,

            lambda x, row, j:
            x[COLNAMES[0]].get(row[0], 0)**2 if j in range(8) else 0,
        ],
    ]

    feature_generator = feature_lists[feature_set]

    # create feature vectors
    logger.debug("creating feature vectors")
    features_train = []
    for row in X_train:
        features_train.append([])
        for j in range(len(COLNAMES)):
            for feature in feature_generator:
                feature_row = feature(dictionaries[j][row[j]], row, j)
                features_train[-1].append(feature_row)
    features_train = np.array(features_train)

    features_test = []
    for row in X_test:
        features_test.append([])
        for j in range(len(COLNAMES)):
            for feature in feature_generator:
                feature_row = feature(dictionaries[j][row[j]], row, j)
                features_test[-1].append(feature_row)
    features_test = np.array(features_test)

    return features_train, features_test


def pre_process(features_train, features_test,
                create_divs=False, log_transform=False, normalize=True):
    """
    Take lists of feature columns as input, pre-process them (eventually
    performing some transformation), then return nicely formatted numpy arrays.
    """
    logger.info("performing preprocessing")

    features_train = list(features_train.T)
    features_test = list(features_test.T)
    features_train = [list(feature) for feature in features_train]
    features_test = [list(feature) for feature in features_test]

    # remove constant features
    for i in range(len(features_train) - 1, -1, -1):
        if np.var(features_train[i]) + np.var(features_test[i]) == 0:
            features_train.pop(i)
            features_test.pop(i)
    n_features = len(features_train)

    # create some polynomial features
    if create_divs:
        for i in range(n_features):
            for j in range(1):
                features_train.append([round(a/(b + 1), 3) for a, b in zip(
                    features_train[i], features_train[j])])
                features_test.append([round(a/(b + 1), 3) for a, b in zip(
                    features_test[i], features_test[j])])

                features_train.append([round(a/(b + 1), 3) for a, b in zip(
                    features_train[j], features_train[i])])
                features_test.append([round(a/(b + 1), 3) for a, b in zip(
                    features_test[j], features_test[i])])

                features_train.append([a*b for a, b in zip(
                    features_train[j], features_train[i])])
                features_test.append([a*b for a, b in zip(
                    features_test[j], features_test[i])])

    if log_transform:
        tmp_train = []
        tmp_test = []
        for i in range(n_features):
            tmp_train.append([math.log(a + 1) if (a + 1) > 0 else 0
                             for a in features_train[i]])
            tmp_test.append([math.log(a + 1) if (a + 1) > 0 else 0
                             for a in features_test[i]])

            tmp_train.append([a**2 for a in features_train[i]])
            tmp_test.append([a**2 for a in features_test[i]])
            tmp_train.append([a**3 for a in features_train[i]])
            tmp_test.append([a**3 for a in features_test[i]])
        features_train = tmp_train
        features_test = tmp_test

    logger.info("created %d features", len(features_train))
    features_train = np.array(features_train).T
    features_test = np.array(features_test).T

    # normalize the new features
    if normalize:
        normalizer = preprocessing.StandardScaler()
        normalizer.fit(features_train)
        features_train = normalizer.transform(features_train)
        features_test = normalizer.transform(features_test)

    return features_train, features_test


def get_pivottable(X_train, X_test, use='all'):
    """
    Returns a list of dictionaries, one per feature in the
    basic data, containing cross-tabulated counts
    for each column and each value of the feature.
    """
    dictionaries = []
    if use == 'all':
        X = np.vstack((X_train, X_test))
        filename = "pivottable"
    elif use == 'train':
        X = X_train
        filename = "pivottable_train"
    else:
        X = X_test
        filename = "pivottable_test"

    for i in range(len(COLNAMES)):
        dictionaries.append({'total': 0})

    try:
        with open("cache/%s.pkl" % filename, 'rb') as f:
            logger.debug("loading cross-tabulated data from cache")
            dictionaries = pickle.load(f)
    except IOError:
        logger.debug("no cache found, cross-tabulating data")
        for i, row in enumerate(X):
            for j in range(len(COLNAMES)):
                dictionaries[j]['total'] += 1
                if row[j] not in dictionaries[j]:
                    dictionaries[j][row[j]] = {'total': 1}
                    for k, key in enumerate(COLNAMES):
                        dictionaries[j][row[j]][key] = {row[k]: 1}
                else:
                    dictionaries[j][row[j]]['total'] += 1
                    for k, key in enumerate(COLNAMES):
                        if row[k] not in dictionaries[j][row[j]][key]:
                            dictionaries[j][row[j]][key][row[k]] = 1
                        else:
                            dictionaries[j][row[j]][key][row[k]] += 1
        with open("cache/%s.pkl" % filename, 'wb') as f:
            pickle.dump(dictionaries, f, pickle.HIGHEST_PROTOCOL)

    return dictionaries


def create_tuples(X):
    logger.debug("creating feature tuples")
    cols = []
    for i in range(X.shape[1]):
        for j in range(i, X.shape[1]):
            cols.append(X[:, i] + X[:, j]*3571)
    return np.hstack((X, np.vstack(cols).T))


def create_triples(X):
    logger.debug("creating feature triples")
    cols = []
    for i in range(X.shape[1]):
        for j in range(i, X.shape[1]):
            for k in range(j, X.shape[1]):
                cols.append(X[:, i]*3461 + X[:, j]*5483 + X[:, k])
    return np.hstack((X, np.vstack(cols).T))


def consolidate(X_train, X_test):
    """
    Transform in-place the given dataset by consolidating
    rare features into a single category.
    """
    X = np.vstack((X_train, X_test))
    relabeler = preprocessing.LabelEncoder()

    for j in range(X.shape[1]):
        relabeler.fit(X[:, j])
        X[:, j] = relabeler.transform(X[:, j])
        X_train[:, j] = relabeler.transform(X_train[:, j])
        X_test[:, j] = relabeler.transform(X_test[:, j])

        raw_counts = np.bincount(X[:, j])
        indices = np.nonzero(raw_counts)[0]
        counts = dict((x, raw_counts[x]) for x in indices)
        max_value = np.max(X[:, j])

        for i in range(X_train.shape[0]):
            if counts[X_train[i, j]] <= 1:
                X_train[i, j] = max_value + 1

        for i in range(X_test.shape[0]):
            if counts[X_test[i, j]] <= 1:
                X_test[i, j] = max_value + 1


class OneHotEncoder():
    """
    OneHotEncoder takes data matrix with categorical columns and
    converts it to a sparse binary matrix.
    """
    def __init__(self):
        self.keymap = None

    def fit(self, X):
        self.keymap = []
        for col in X.T:
            uniques = set(list(col))
            self.keymap.append(dict((key, i) for i, key in enumerate(uniques)))

    def transform(self, X):
        if self.keymap is None:
            self.fit(X)

        outdat = []
        for i, col in enumerate(X.T):
            km = self.keymap[i]
            num_labels = len(km)
            spmat = sparse.lil_matrix((X.shape[0], num_labels))
            for j, val in enumerate(col):
                if val in km:
                    spmat[j, km[val]] = 1
            outdat.append(spmat)
        outdat = sparse.hstack(outdat).tocsr()
        return outdat
