Amazon Employee Access Challenge
================================

This code was written by Paul Duan (<email@paulduan.com>) and Benjamin Solecki (<bensolucky@gmail.com>).
It provides our winning solution to the Amazon Employee Access Challenge.
Our code is currently not merged. You'll find Benjamin's code in the BSMan/ folder, which needs to be run separately.


Usage:
---------------
    [python] classifier.py [-h] [-d] [-i ITER] [-f OUTPUTFILE] [-g] [-m] [-n] [-s] [-v] [-w]

    Parameters for the script.

    optional arguments:
      -h, --help            show this help message and exit
      -d, --diagnostics     Compute diagnostics.
      -i ITER, --iter ITER  Number of iterations for averaging.
      -f OUTPUTFILE, --outputfile OUTPUTFILE
                            Name of the file where predictions are saved.
      -g, --grid-search     Use grid search to find best parameters.
      -m, --model-selection
                            Use model selection.
      -n, --no-cache        Use cache.
      -s, --stack           Use stacking.
      -v, --verbose         Show computation steps.
      -w, --fwls            Use metafeatures.


To directly generate predictions on the test set without computing CV
metrics, simply run:  

    python classifier.py -i0 -f[output_filename]

This script will launch Paul's model, which incorporates some of Benjamin's features.
Benjamin's model is in the BSMan folder and can be run this way:  

    (in BSMan/)
    [python] logistic.py log 75
    [python] ensemble.py

The output of our models is then combined by simple standardization then weighted averaging, using 2/3 Paul's model and 1/3 Benjamin's.


Requirements:
---------------
This code requires Python, numpy/scipy, scikit-learn, and pandas for
some of the external code (this dependency will be removed in the
future).  
It has been tested under Mac OS X with Python v.7.x,
scikit-learn 0.13, numpy 0.17, and pandas 0.11.

License:
---------------
This content is released under the [MIT Licence](http://opensource.org/licenses/MIT).
