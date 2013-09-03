#!/usr/bin/env python

"""Amazon Access Challenge

This is my part of the code that produced the winning solution to the
Amazon Employee Access Challenge. See README.md for more details.

Author: Paul Duan <email@paulduan.com>
"""

from __future__ import division

import argparse
import logging

from sklearn import metrics, cross_validation, linear_model, ensemble
from helpers import ml, diagnostics
from helpers.data import load_data, save_results
from helpers.feature_extraction import create_datasets

logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                    filename="history.log", filemode='a', level=logging.DEBUG,
                    datefmt='%m/%d/%y %H:%M:%S')
formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                              datefmt='%m/%d/%y %H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

logger = logging.getLogger(__name__)


def main(CONFIG):
    """
    The final model is a combination of several base models, which are then
    combined using StackedClassifier defined in the helpers.ml module.

    The list of models and associated datasets is generated automatically
    from their identifying strings. The format is as follows:
    A:b_c where A is the initials of the algorithm to use, b is the base
    dataset, and c is the feature set and the variants to use.
    """
    SEED = 42
    selected_models = [
        "LR:tuples_sf",
        "LR:greedy_sfl",
        "LR:greedy2_sfl",
        "LR:greedy3_sf",
        "RFC:basic_b",
        "RFC:tuples_f",
        "RFC:tuples_fd",
        "RFC:greedy_f",
        "RFC:greedy2_f",
        "GBC:basic_f",
        "GBC:tuples_f",
        "LR:greedy_sbl",
        "GBC:greedy_c",
        "GBC:tuples_cf",
        #"RFC:effects_f",  # experimental; added after the competition
    ]

    # Create the models on the fly
    models = []
    for item in selected_models:
        model_id, dataset = item.split(':')
        model = {'LR': linear_model.LogisticRegression,
                 'GBC': ensemble.GradientBoostingClassifier,
                 'RFC': ensemble.RandomForestClassifier,
                 'ETC': ensemble.ExtraTreesClassifier}[model_id]()
        model.set_params(random_state=SEED)
        models.append((model, dataset))

    datasets = [dataset for model, dataset in models]

    logger.info("loading data")
    y, X = load_data('train.csv')
    X_test = load_data('test.csv', return_labels=False)

    logger.info("preparing datasets (use_cache=%s)", str(CONFIG.use_cache))
    create_datasets(X, X_test, y, datasets, CONFIG.use_cache)

    # Set params
    for model, feature_set in models:
        model.set_params(**ml.find_params(model, feature_set, y,
                                          grid_search=CONFIG.grid_search))
    clf = ml.StackedClassifier(
        models, stack=CONFIG.stack, fwls=CONFIG.fwls,
        model_selection=CONFIG.model_selection,
        use_cached_models=CONFIG.use_cache)

    #  Metrics
    logger.info("computing cv score")
    mean_auc = 0.0
    for i in range(CONFIG.iter):
        train, cv = cross_validation.train_test_split(
            range(len(y)), test_size=.20, random_state=1+i*SEED)
        cv_preds = clf.fit_predict(y, train, cv, show_steps=CONFIG.verbose)

        fpr, tpr, _ = metrics.roc_curve(y[cv], cv_preds)
        roc_auc = metrics.auc(fpr, tpr)
        logger.info("AUC (fold %d/%d): %.5f", i + 1, CONFIG.iter, roc_auc)
        mean_auc += roc_auc

        if CONFIG.diagnostics and i == 0:  # only plot for first fold
            logger.info("plotting learning curve")
            diagnostics.learning_curve(clf, y, train, cv)
            diagnostics.plot_roc(fpr, tpr)
    if CONFIG.iter:
        logger.info("Mean AUC: %.5f",  mean_auc/CONFIG.iter)

    # Create submissions
    if CONFIG.outputfile:
        logger.info("making test submissions (CV AUC: %.4f)", mean_auc)
        preds = clf.fit_predict(y, show_steps=CONFIG.verbose)
        save_results(preds, CONFIG.outputfile + ".csv")

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="Parameters for the script.")
    PARSER.add_argument('-d', "--diagnostics", action="store_true",
                        help="Compute diagnostics.")
    PARSER.add_argument('-i', "--iter", type=int, default=1,
                        help="Number of iterations for averaging.")
    PARSER.add_argument("-f", "--outputfile", default="",
                        help="Name of the file where predictions are saved.")
    PARSER.add_argument('-g', "--grid-search", action="store_true",
                        help="Use grid search to find best parameters.")
    PARSER.add_argument('-m', "--model-selection", action="store_true",
                        default=False, help="Use model selection.")
    PARSER.add_argument('-n', "--no-cache", action="store_false", default=True,
                        help="Use cache.", dest="use_cache")
    PARSER.add_argument("-s", "--stack", action="store_true",
                        help="Use stacking.")
    PARSER.add_argument('-v', "--verbose", action="store_true",
                        help="Show computation steps.")
    PARSER.add_argument("-w", "--fwls", action="store_true",
                        help="Use metafeatures.")
    PARSER.set_defaults(argument_default=False)
    CONFIG = PARSER.parse_args()

    CONFIG.stack = CONFIG.stack or CONFIG.fwls

    logger.debug('\n' + '='*50)
    main(CONFIG)
