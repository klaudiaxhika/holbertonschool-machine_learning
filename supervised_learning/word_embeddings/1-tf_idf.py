#!/usr/bin/env python3

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def tf_idf(sentences, vocab=None):
    if vocab is None:
        vocab = set(word for sentence in sentences for word in sentence.split())
    
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names()
    
    return embeddings, features
