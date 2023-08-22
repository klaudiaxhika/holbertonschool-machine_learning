#!/usr/bin/env python3

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words(sentences, vocab=None):
    if vocab is None:
        vocab = set(word for sentence in sentences for word in sentence.split())
    
    vectorizer = CountVectorizer(vocabulary=vocab)
    embeddings = vectorizer.transform(sentences).toarray()
    features = vectorizer.get_feature_names()
    
    return embeddings, features
