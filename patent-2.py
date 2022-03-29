import numpy as np
import pandas as pd
import xgboost as xgb
import spacy

nlp = spacy.load("en_core_web_lg")

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train.head()
train.context.unique().shape
train.context.unique().shape
test.context.unique().shape

train.anchor.unique().shape

train_anchor_spacy = [nlp(row).vector for row in train.anchor]
train_target_spacy = [nlp(row).vector for row in train.target]
train_anchor_spacy = np.asarray(train_anchor_spacy)
train_target_spacy = np.asarray(train_target_spacy)

train_spacy = np.column_stack((train_anchor_spacy, train_target_spacy))
train_spacy.shape

test_anchor_spacy = [nlp(row).vector for row in test.anchor]
test_target_spacy = [nlp(row).vector for row in test.target]
test_anchor_spacy = np.asarray(test_anchor_spacy)
test_target_spacy = np.asarray(test_target_spacy)

test_spacy = np.column_stack((test_anchor_spacy, test_target_spacy))
test_spacy.shape

data = train_spacy
label = train.score
dtrain = xgb.DMatrix(data, label=label)

xgb.get_config()

param = {'max_depth':12, 
         'learning_rate':0.1,
         'verbosity':2,
         'objective':'binary:logistic',
         'eval_metric': ['logloss', 'aucpr', 'auc']}
num_rounds = 250

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_spacy, train.score, test_size=0.33, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

bst = xgb.train(param, dtrain, num_rounds, evals=watchlist,
                early_stopping_rounds=25, verbose_eval=True)

bst.best_score
bst.best_iteration

np.unique(train.context, return_counts=True)
np.shape(np.unique(train.context, return_counts=True))

bst_cv = xgb.cv(param, dtrain, num_rounds, nfold=5, #evals=watchlist,
                early_stopping_rounds=25, verbose_eval=True)
print(bst_cv['test-logloss-mean'])
nrounds = round(bst_cv.shape[0]/.8)

#from sklearn.model_selection import RandomizedSearchCV, KFold
dtest = xgb.DMatrix(test_spacy)
ypred = bst.predict(dtest, iteration_range=(0, bst.best_iteration))