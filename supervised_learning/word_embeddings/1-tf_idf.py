#!/usr/bin/env python3
"""
Creates a TF-IDF embedding
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding
    sentences: a list of sentences to analyze
    vocab: a list of the vocabulary words to use for the analysis
        If None, all words within sentences should be used
    """
    tfidf = TfidfVectorizer(vocabulary=vocab)
    X = tfidf.fit_transform(sentences)
    features = tfidf.get_feature_names()
    embeddings = X.toarray()

    return embeddings, features
