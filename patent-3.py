import numpy as np
import pandas as pd
import xgboost as xgb
from transformers import BertModel, BertTokenizer, TFBertModel

import torch
import tensorflow as tf

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = TFBertModel.from_pretrained("bert-large-uncased")

x_tf = tf.math.reduce_mean(model(tokenizer(train.anchor[0], return_tensors='tf'))[0], axis=1).numpy()

#model(tokenizer(train.anchor[0], return_tensors='tf'))[0]
#tf.math.reduce_mean(model(tokenizer(train.anchor[0], return_tensors='tf'))[0], axis=1).shape
#tf.math.reduce_mean(model(tokenizer(train.anchor[0], return_tensors='tf'))[0], axis=1).numpy()

train_anchor_bert = [tf.math.reduce_mean(model(tokenizer(row, return_tensors='tf'))[0], axis=1).numpy() for row in train.anchor]
train_target_bert = [tf.math.reduce_mean(model(tokenizer(row, return_tensors='tf'))[0], axis=1).numpy() for row in train.target]
train_bert = np.column_stack((train_anchor_bert[0], train_target_bert[0]))
train_bert.shape

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained("bert-large-uncased")

x_pt = torch.mean(model(**tokenizer(train.anchor[0], return_tensors='pt'))[0], axis=1).detach().numpy()

#model(**tokenizer(train.anchor[0], return_tensors='pt'))[0]
#torch.mean(model(**tokenizer(train.anchor[0], return_tensors='pt'))[0], axis=1).shape
#torch.mean(model(**tokenizer(train.anchor[0], return_tensors='pt'))[0], axis=1).detach().numpy()

train_anchor_bert = [torch.mean(model(**tokenizer(row, return_tensors='pt'))[0], axis=1).detach().numpy() for row in train.anchor]
train_target_bert = [torch.mean(model(**tokenizer(row, return_tensors='pt'))[0], axis=1).detach().numpy() for row in train.target]
train_bert = np.column_stack((train_anchor_bert[0], train_target_bert[0]))
train_bert.shape

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
