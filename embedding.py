import torch
import numpy as np
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased')


tokenizer.tokenize("Apple feel from the tree")
tokens = tokenizer.tokenize("Apple feel from the tree")
tokenizer.covert_tokens_to_ids(tokens)

tokens = tokenizer("Apple feel from the tree", return_tensors='pt', add_special_tokens=True)
tokenizer.decode(tokens['input_ids'][0])

with torch.no_grad():
    last_hidden_states = model(**tokens)
    print(last_hidden_states)
last_hidden_states[0].shape
torch.mean(last_hidden_states[0], 1).shape
torch.mean(last_hidden_states[0])
torch.mean(last_hidden_states[0]).numpy()

def cosine_similarity(a,b):
    return a.dot(b)/np.sqrt(a.dot(a) * b.dot(b))

#torch.index_select(last_hidden_states[0], 1, torch.tensor[2])

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
tokenizer.decode(encoded_input['input_ids'][0])
output = model(**encoded_input)
output[0].shape

from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = TFBertModel.from_pretrained("bert-large-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
tokenizer.decode(encoded_input['input_ids'][0])
output = model(encoded_input)
output[0].shape

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModel, AutoModelForPreTraining

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForPreTraining.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModel.from_pretrained("EleutherAI/gpt-j-6B")

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
tokenizer.decode(encoded_input['input_ids'][0])
output = model(**encoded_input)
output[0].shape

import spacy
nlp = spacy.load("en_core_web_lg")
nlp("Apple feel from the tree").vector.shape
cosine_similarity(nlp('dog').vector, nlp('dog').vector)
cosine_similarity(nlp('Apple feel from the tree').vector, nlp('Only make juice with red apples').vector)
nlp('Apple feel from the tree').vector.shape

torch.get_num_threads()
torch.backends.mkl.is_available()
torch.backends.openmp.is_available()