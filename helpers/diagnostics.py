"""diagnostics.py

Some methods to plot diagnostics.

Author: Paul Duan <email@paulduan.com>
"""

import matplotlib.pyplot as plt
from sklearn.metrics import hinge_loss


def plot_roc(fpr, tpr):
    """Plot ROC curve and display it."""
    plt.clf()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')


def learning_curve(classifier, y, train, cv, n=15):
    """Plot train and cv loss for increasing train sample sizes."""
    chunk = int(len(y)/n)
    n_samples = []
    train_losses = []
    cv_losses = []
    previous_cache_dir = classifier.cache_dir
    classifier.cache_dir = "diagnostics"

    for i in range(n):
        train_subset = train[:(i + 1)*chunk]
        preds_cv = classifier.fit_predict(y, train_subset, cv,
                                          show_steps=False)
        preds_train = classifier.fit_predict(y, train_subset, train_subset,
                                             show_steps=False)
        n_samples.append((i + 1)*chunk)
        cv_losses.append(hinge_loss(y[cv], preds_cv, neg_label=0))
        train_losses.append(hinge_loss(y[train_subset], preds_train,
                            neg_label=0))

    classifier.cache_dir = previous_cache_dir
    plt.clf()
    plt.plot(n_samples, train_losses, 'r--', n_samples, cv_losses, 'b--')
    plt.ylim([min(train_losses) - .01, max(cv_losses) + .01])

    plt.savefig('plots/learning_curve.png')
    plt.show()
