"""ml.py

This is the file that does the heavy lifting.
It contains the ML algorithms themselves:
    - AUCRegressor: a custom class that optimizes AUC directly
    - MLR: a linear regression with non-negativity constraints
    - StackedClassifier: a custom class that combines several models

And some related functions:
    - find_params: sets the hyperparameters for a given model

Author: Paul Duan <email@paulduan.com>
"""

from __future__ import division

import cPickle as pickle
import itertools
import json
import logging
import multiprocessing
import scipy as sp
import numpy as np

from functools import partial
from operator import itemgetter

from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, linear_model

from data import load_from_cache, get_dataset
from utils import stringify, compute_auc

logger = logging.getLogger(__name__)

N_TREES = 500

INITIAL_PARAMS = {
    'LogisticRegression': {'C': 2, 'penalty': 'l2', 'class_weight': 'auto'},
    'RandomForestClassifier': {
        'n_estimators': N_TREES, 'n_jobs': 4,
        'min_samples_leaf': 2, 'bootstrap': False,
        'max_depth': 30, 'min_samples_split': 5, 'max_features': .1
    },
    'ExtraTreesClassifier': {
        'n_estimators': N_TREES, 'n_jobs': 3, 'min_samples_leaf': 2,
        'max_depth': 30, 'min_samples_split': 5, 'max_features': .1,
        'bootstrap': False,
    },
    'GradientBoostingClassifier': {
        'n_estimators': N_TREES, 'learning_rate': .08, 'max_features': 7,
        'min_samples_leaf': 1, 'min_samples_split': 3, 'max_depth': 5,
    },
}

PARAM_GRID = {
    'LogisticRegression': {'C': [1.5, 2, 2.5, 3, 3.5, 5, 5.5],
                           'class_weight': ['auto']},
    'RandomForestClassifier': {
        'n_jobs': [1], 'max_depth': [15, 20, 25, 30, 35, None],
        'min_samples_split': [1, 3, 5, 7],
        'max_features': [3, 8, 11, 15],
    },
    'ExtraTreesClassifier': {'min_samples_leaf': [2, 3],
                             'n_jobs': [1],
                             'min_samples_split': [1, 2, 5],
                             'bootstrap': [False],
                             'max_depth': [15, 20, 25, 30],
                             'max_features': [1, 3, 5, 11]},
    'GradientBoostingClassifier': {'max_features': [4, 5, 6, 7],
                                   'learning_rate': [.05, .08, .1],
                                   'max_depth': [8, 10, 13]},
}


class AUCRegressor(object):
    def __init__(self):
        self.coef_ = 0

    def _auc_loss(self, coef, X, y):
        fpr, tpr, _ = roc_curve(y, sp.dot(X, coef))
        return -auc(fpr, tpr)

    def fit(self, X, y):
        lr = linear_model.LinearRegression()
        auc_partial = partial(self._auc_loss, X=X, y=y)
        initial_coef = lr.fit(X, y).coef_
        self.coef_ = sp.optimize.fmin(auc_partial, initial_coef)

    def predict(self, X):
        return sp.dot(X, self.coef_)

    def score(self, X, y):
        fpr, tpr, _ = roc_curve(y, sp.dot(X, self.coef_))
        return auc(fpr, tpr)


class MLR(object):
    def __init__(self):
        self.coef_ = 0

    def fit(self, X, y):
        self.coef_ = sp.optimize.nnls(X, y)[0]
        self.coef_ = np.array(map(lambda x: x/sum(self.coef_), self.coef_))

    def predict(self, X):
        predictions = np.array(map(sum, self.coef_ * X))
        return predictions

    def score(self, X, y):
        fpr, tpr, _ = roc_curve(y, sp.dot(X, self.coef_))
        return auc(fpr, tpr)


class StackedClassifier(object):
    """
    Implement stacking to combine several models.
    The base (stage 0) models can be either combined through
    simple averaging (fastest), or combined using a stage 1 generalizer
    (requires computing CV predictions on the train set).

    See http://ijcai.org/Past%20Proceedings/IJCAI-97-VOL2/PDF/011.pdf:
    "Stacked generalization: when does it work?", Ting and Witten, 1997

    For speed and convenience, both fitting and prediction are done
    in the same method fit_predict; this is done in order to enable
    one to compute metrics on the predictions after training each model without
    having to wait for all the models to be trained.

    Options:
    ------------------------------
    - models: a list of (model, dataset) tuples that represent stage 0 models
    - generalizer: an Estimator object. Must implement fit and predict
    - model_selection: boolean. Whether to use brute force search to find the
        optimal subset of models that produce the best AUC.
    """
    def __init__(self, models, generalizer=None, model_selection=True,
                 stack=False, fwls=False, use_cached_models=True):
        self.cache_dir = "main"
        self.models = models
        self.model_selection = model_selection
        self.stack = stack
        self.fwls = fwls
        self.generalizer = linear_model.RidgeCV(
            alphas=np.linspace(0, 200), cv=100)
        self.use_cached_models = use_cached_models

    def _combine_preds(self, X_train, X_cv, y, train=None, predict=None,
                       stack=False, fwls=False):
        """
        Combine preds, returning in order:
            - mean_preds: the simple average of all model predictions
            - stack_preds: the predictions of the stage 1 generalizer
            - fwls_preds: same as stack_preds, but optionally using more
                complex blending schemes (meta-features, different
                generalizers, etc.)
        """
        mean_preds = np.mean(X_cv, axis=1)
        stack_preds = None
        fwls_preds = None

        if stack:
            self.generalizer.fit(X_train, y)
            stack_preds = self.generalizer.predict(X_cv)

        if self.fwls:
            meta, meta_cv = get_dataset('metafeatures', train, predict)
            fwls_train = np.hstack((X_train, meta))
            fwls_cv = np.hstack((X_cv, meta))
            self.generalizer.fit(fwls_train)
            fwls_preds = self.generalizer.predict(fwls_cv)

        return mean_preds, stack_preds, fwls_preds

    def _find_best_subset(self, y, predictions_list):
        """Finds the combination of models that produce the best AUC."""
        best_subset_indices = range(len(predictions_list))

        pool = multiprocessing.Pool(processes=4)
        partial_compute_subset_auc = partial(compute_subset_auc,
                                             pred_set=predictions_list, y=y)
        best_auc = 0
        best_n = 0
        best_indices = []

        if len(predictions_list) == 1:
            return [1]

        for n in range(int(len(predictions_list)/2), len(predictions_list)):
            cb = itertools.combinations(range(len(predictions_list)), n)
            combination_results = pool.map(partial_compute_subset_auc, cb)
            best_subset_auc, best_subset_indices = max(
                combination_results, key=itemgetter(0))
            print "- best subset auc (%d models): %.4f > %s" % (
                n, best_subset_auc, n, list(best_subset_indices))
            if best_subset_auc > best_auc:
                best_auc = best_subset_auc
                best_n = n
                best_indices = list(best_subset_indices)
        pool.terminate()

        logger.info("best auc: %.4f", best_auc)
        logger.info("best n: %d", best_n)
        logger.info("best indices: %s", best_indices)
        for i, (model, feature_set) in enumerate(self.models):
            if i in best_subset_indices:
                logger.info("> model: %s (%s)", model.__class__.__name__,
                            feature_set)

        return best_subset_indices

    def _get_model_preds(self, model, X_train, X_predict, y_train, cache_file):
        """
        Return the model predictions on the prediction set,
        using cache if possible.
        """
        model_output = load_from_cache(
            "models/%s/%s.pkl" % (self.cache_dir, cache_file),
            self.use_cached_models)

        model_params, model_preds = model_output \
            if model_output is not None else (None, None)

        if model_preds is None or model_params != model.get_params():
            model.fit(X_train, y_train)
            model_preds = model.predict_proba(X_predict)[:, 1]
            with open("cache/models/%s/%s.pkl" % (
                    self.cache_dir, cache_file), 'wb') as f:
                pickle.dump((model.get_params(), model_preds), f)

        return model_preds

    def _get_model_cv_preds(self, model, X_train, y_train, cache_file):
        """
        Return cross-validation predictions on the training set, using cache
        if possible.
        This is used if stacking is enabled (ie. a second model is used to
        combine the stage 0 predictions).
        """
        stack_preds = load_from_cache(
            "models/%s/cv_preds/%s.pkl" % (self.cache_dir, cache_file),
            self.use_cached_models)

        if stack_preds is None:
            kfold = cross_validation.StratifiedKFold(y_train, 4)
            stack_preds = []
            indexes_cv = []
            for stage0, stack in kfold:
                model.fit(X_train[stage0], y_train[stage0])
                stack_preds.extend(list(model.predict_proba(
                    X_train[stack])[:, 1]))
                indexes_cv.extend(list(stack))
            stack_preds = np.array(stack_preds)[sp.argsort(indexes_cv)]

            with open("cache/models/%s/cv_preds/%s%d.pkl" % (
                    self.cache_dir, cache_file), 'wb') as f:
                pickle.dump(stack_preds, f, pickle.HIGHEST_PROTOCOL)

        return stack_preds

    def fit_predict(self, y, train=None, predict=None, show_steps=True):
        """
        Fit each model on the appropriate dataset, then return the average
        of their individual predictions. If train is specified, use a subset
        of the training set to train the models, then predict the outcome of
        either the remaining samples or (if given) those specified in cv.
        If train is omitted, train the models on the full training set, then
        predict the outcome of the full test set.

        Options:
        ------------------------------
        - y: numpy array. The full vector of the ground truths.
        - train: list. The indices of the elements to be used for training.
            If None, take the entire training set.
        - predict: list. The indices of the elements to be predicted.
        - show_steps: boolean. Whether to compute metrics after each stage
            of the computation.
        """
        y_train = y[train] if train is not None else y
        if train is not None and predict is None:
            predict = [i for i in range(len(y)) if i not in train]

        stage0_train = []
        stage0_predict = []
        for model, feature_set in self.models:
            X_train, X_predict = get_dataset(feature_set, train, predict)

            identifier = train[0] if train is not None else -1
            cache_file = stringify(model, feature_set) + str(identifier)

            model_preds = self._get_model_preds(
                model, X_train, X_predict, y_train, cache_file)
            stage0_predict.append(model_preds)

            # if stacking, compute cross-validated predictions on the train set
            if self.stack:
                model_cv_preds = self._get_model_cv_preds(
                    model, X_train, y_train, cache_file)
                stage0_train.append(model_cv_preds)

            # verbose mode: compute metrics after every model computation
            if show_steps:
                if train is not None:
                    mean_preds, stack_preds, fwls_preds = self._combine_preds(
                        np.array(stage0_train).T, np.array(stage0_predict).T,
                        y_train, train, predict,
                        stack=self.stack, fwls=self.fwls)

                    model_auc = compute_auc(y[predict], stage0_predict[-1])
                    mean_auc = compute_auc(y[predict], mean_preds)
                    stack_auc = compute_auc(y[predict], stack_preds) \
                        if self.stack else 0
                    fwls_auc = compute_auc(y[predict], fwls_preds) \
                        if self.fwls else 0

                    logger.info(
                        "> AUC: %.4f (%.4f, %.4f, %.4f) [%s]", model_auc,
                        mean_auc, stack_auc, fwls_auc,
                        stringify(model, feature_set))
                else:
                    logger.info("> used model %s:\n%s", stringify(
                        model, feature_set), model.get_params())

        if self.model_selection and predict is not None:
            best_subset = self._find_best_subset(y[predict], stage0_predict)
            stage0_train = [pred for i, pred in enumerate(stage0_train)
                            if i in best_subset]
            stage0_predict = [pred for i, pred in enumerate(stage0_predict)
                              if i in best_subset]

        mean_preds, stack_preds, fwls_preds = self._combine_preds(
            np.array(stage0_train).T, np.array(stage0_predict).T,
            y_train, stack=self.stack, fwls=self.fwls)

        if self.stack:
            selected_preds = stack_preds if not self.fwls else fwls_preds
        else:
            selected_preds = mean_preds

        return selected_preds


def compute_subset_auc(indices, pred_set, y):
    subset = [vect for i, vect in enumerate(pred_set) if i in indices]
    mean_preds = sp.mean(subset, axis=0)
    mean_auc = compute_auc(y, mean_preds)

    return mean_auc, indices


def find_params(model, feature_set, y, subsample=None, grid_search=False):
    """
    Return parameter set for the model, either predefined
    or found through grid search.
    """
    model_name = model.__class__.__name__
    params = INITIAL_PARAMS.get(model_name, {})
    y = y if subsample is None else y[subsample]

    try:
        with open('saved_params.json') as f:
            saved_params = json.load(f)
    except IOError:
        saved_params = {}

    if (grid_search and model_name in PARAM_GRID and stringify(
            model, feature_set) not in saved_params):
        X, _ = get_dataset(feature_set, subsample, [0])
        clf = GridSearchCV(model, PARAM_GRID[model_name], cv=10, n_jobs=6,
                           scoring="roc_auc")
        clf.fit(X, y)
        logger.info("found params (%s > %.4f): %s",
                    stringify(model, feature_set),
                    clf.best_score_, clf.best_params_)
        params.update(clf.best_params_)
        saved_params[stringify(model, feature_set)] = params
        with open('saved_params.json', 'w') as f:
            json.dump(saved_params, f, indent=4, separators=(',', ': '),
                      ensure_ascii=True, sort_keys=True)
    else:
        params.update(saved_params.get(stringify(model, feature_set), {}))
        if grid_search:
            logger.info("using params %s: %s", stringify(model, feature_set),
                        params)

    return params
