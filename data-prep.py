import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

def wordvec(x1, x2):
    m1 = model(**tokenizer(x1, return_tensors='pt'))
    m2 = model(**tokenizer(x2, return_tensors='pt'))
    n1 = torch.mean(m1[0], axis=1).detach().numpy()
    n2 = torch.mean(m1[0], axis=1).detach().numpy()
    return np.concatenate((n1[0], n2[0]), axis=0)

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
model = BertModel.from_pretrained("bert-large-uncased")

train_bert = np.array([wordvec(i,j) for i,j in zip(train.anchor, train.target)])
train_bert.shape

df = pd.DataFrame(train_bert)
df.rename(columns=lambda x: 'x'+str(x), inplace=True)
df.to_parquet('train_bert.parquet', compression='zstd')

tokenizer = AutoTokenizer.from_pretrained("anferico/bert-for-patents")
model = AutoModel.from_pretrained("anferico/bert-for-patents")

train_pat = np.array([wordvec(i,j) for i,j in zip(train.anchor, train.target)])
train_pat.shape

df = pd.DataFrame(train_pat)
df.rename(columns=lambda x: 'x'+str(x), inplace=True)
df.to_parquet('train_pat.parquet', compression='zstd')