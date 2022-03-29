import tokenize
import numpy as np
import pandas as pd
import sys

print(np.__version__)
print(pd.__version__)
print(sys.version)

import spacy
#Components: tok2vec, tagger, parser, senter, ner, attribute_ruler, lemmatizer
#spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")

import sklearn
sklearn.show_versions()

from transformers import pipeline
import tensorflow as tf
print(tf.__version__)
print(tf.version.VERSION)   

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
tf.config.list_physical_devices("GPU")

from transformers import BertModel, BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased')

# Process whole documents
text = ("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")
doc = nlp(text)

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)