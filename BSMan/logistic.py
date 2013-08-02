"""
This program is based on code submitted by Miroslaw Horbal to the Kaggle 
forums, which was itself based on an earlier submission from Paul Doan.
My thanks to both.

Author: Benjamin Solecki <bensolucky@gmail.com>
"""

from numpy import array, hstack
from sklearn import metrics, cross_validation, linear_model
from sklearn import naive_bayes
from sklearn import preprocessing
from scipy import sparse
from itertools import combinations

from sets import Set
import numpy as np
import pandas as pd
import sys

#SEED = 55
SEED = int(sys.argv[2])

def group_data(data, degree=3, hash=hash):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m,n = data.shape
    for indicies in combinations(range(n), degree):
	if 5 in indicies and 7 in indicies:
	    print "feature Xd"
	elif 2 in indicies and 3 in indicies:
	    print "feature Xd"
	else:
            new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return array(new_data).T

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

def create_test_submission(filename, prediction):
    content = ['id,ACTION']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(i+1,p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved'

# This loop essentially from Paul's starter code
# I (Ben) increased the size of train at the expense of test, because
# when train is small many features will not be found in train.
def cv_loop(X, y, model, N):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                                       X, y, test_size=1.0/float(N), 
                                       random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:,1]
        auc = metrics.auc_score(y_cv, preds)
        #print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
        mean_auc += auc
    return mean_auc/N
    
learner = sys.argv[1]
print "Reading dataset..."
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
submit=learner + str(SEED) + '.csv'
all_data = np.vstack((train_data.ix[:,1:-1], test_data.ix[:,1:-1]))
num_train = np.shape(train_data)[0]

# Transform data
print "Transforming data..."
# Relabel the variable values to smallest possible so that I can use bincount
# on them later.
relabler = preprocessing.LabelEncoder()
for col in range(len(all_data[0,:])):
    relabler.fit(all_data[:, col])
    all_data[:, col] = relabler.transform(all_data[:, col])
########################## 2nd order features ################################
dp = group_data(all_data, degree=2) 
for col in range(len(dp[0,:])):
    relabler.fit(dp[:, col])
    dp[:, col] = relabler.transform(dp[:, col])
    uniques = len(set(dp[:,col]))
    maximum = max(dp[:,col])
    print col
    if maximum < 65534:
        count_map = np.bincount((dp[:, col]).astype('uint16'))
        for n,i in enumerate(dp[:, col]):
            if count_map[i] <= 1:
                dp[n, col] = uniques
            elif count_map[i] == 2:
                dp[n, col] = uniques+1
    else:
        for n,i in enumerate(dp[:, col]):
            if (dp[:, col] == i).sum() <= 1:
                dp[n, col] = uniques
            elif (dp[:, col] == i).sum() == 2:
                dp[n, col] = uniques+1
    print uniques # unique values
    uniques = len(set(dp[:,col]))
    print uniques
    relabler.fit(dp[:, col])
    dp[:, col] = relabler.transform(dp[:, col])
########################## 3rd order features ################################
dt = group_data(all_data, degree=3)
for col in range(len(dt[0,:])):
    relabler.fit(dt[:, col])
    dt[:, col] = relabler.transform(dt[:, col])
    uniques = len(set(dt[:,col]))
    maximum = max(dt[:,col])
    print col
    if maximum < 65534:
        count_map = np.bincount((dt[:, col]).astype('uint16'))
        for n,i in enumerate(dt[:, col]):
            if count_map[i] <= 1:
                dt[n, col] = uniques
            elif count_map[i] == 2:
                dt[n, col] = uniques+1
    else:
        for n,i in enumerate(dt[:, col]):
            if (dt[:, col] == i).sum() <= 1:
                dt[n, col] = uniques
            elif (dt[:, col] == i).sum() == 2:
                dt[n, col] = uniques+1
    print uniques
    uniques = len(set(dt[:,col]))
    print uniques
    relabler.fit(dt[:, col])
    dt[:, col] = relabler.transform(dt[:, col])
########################## 1st order features ################################
for col in range(len(all_data[0,:])):
    relabler.fit(all_data[:, col])
    all_data[:, col] = relabler.transform(all_data[:, col])
    uniques = len(set(all_data[:,col]))
    maximum = max(all_data[:,col])
    print col
    if maximum < 65534:
        count_map = np.bincount((all_data[:, col]).astype('uint16'))
        for n,i in enumerate(all_data[:, col]):
            if count_map[i] <= 1:
                all_data[n, col] = uniques
            elif count_map[i] == 2:
                all_data[n, col] = uniques+1
    else:
        for n,i in enumerate(all_data[:, col]):
            if (all_data[:, col] == i).sum() <= 1:
                all_data[n, col] = uniques
            elif (all_data[:, col] == i).sum() == 2:
                all_data[n, col] = uniques+1
    print uniques
    uniques = len(set(all_data[:,col]))
    print uniques
    relabler.fit(all_data[:, col])
    all_data[:, col] = relabler.transform(all_data[:, col])

# Collect the training features together
y = array(train_data.ACTION)
X = all_data[:num_train]
X_2 = dp[:num_train]
X_3 = dt[:num_train]

# Collect the testing features together
X_test = all_data[num_train:]
X_test_2 = dp[num_train:]
X_test_3 = dt[num_train:]

X_train_all = np.hstack((X, X_2, X_3))
X_test_all = np.hstack((X_test, X_test_2, X_test_3))
num_features = X_train_all.shape[1]
    
if learner == 'NB':
    model = naive_bayes.BernoulliNB(alpha=0.03)
else:
    model = linear_model.LogisticRegression(class_weight='auto', penalty='l2')
    
# Xts holds one hot encodings for each individual feature in memory
# speeding up feature selection 
Xts = [OneHotEncoder(X_train_all[:,[i]])[0] for i in range(num_features)]
    
print "Performing greedy feature selection..."
score_hist = []
N = 10
good_features = set([])
# Greedy feature selection loop
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
print "Selected features %s" % good_features
gf = open("feats" + submit, 'w')
print >>gf, good_features
gf.close()
print len(good_features), " features"
    
print "Performing hyperparameter selection..."
# Hyperparameter selection loop
score_hist = []
Xt = sparse.hstack([Xts[j] for j in good_features]).tocsr()
if learner == 'NB':
    Cvals = [0.001, 0.003, 0.006, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1]
else:
    Cvals = np.logspace(-4, 4, 15, base=2)  # for logistic
for C in Cvals:
    if learner == 'NB':
        model.alpha = C
    else:
        model.C = C
    score = cv_loop(Xt, y, model, N)
    score_hist.append((score,C))
    print "C: %f Mean AUC: %f" %(C, score)
bestC = sorted(score_hist)[-1][1]
print "Best C value: %f" % (bestC)
    
print "Performing One Hot Encoding on entire dataset..."
Xt = np.vstack((X_train_all[:,good_features], X_test_all[:,good_features]))
Xt, keymap = OneHotEncoder(Xt)
X_train = Xt[:num_train]
X_test = Xt[num_train:]
    
if learner == 'NB':
    model.alpha = bestC
else:
    model.C = bestC

print "Training full model..."
print "Making prediction and saving results..."
model.fit(X_train, y)
preds = model.predict_proba(X_test)[:,1]
create_test_submission(submit, preds)
preds = model.predict_proba(X_train)[:,1]
create_test_submission('Train'+submit, preds)
