#!/usr/bin/env python3

from gensim.models import Word2Vec

def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5, cbow=True, iterations=5, seed=0, workers=1):
    
    model = Word2Vec(
        sentences, 
        vector_size=size, 
        min_count=min_count, 
        window=window, 
        negative=negative, 
        sg=0 if cbow else 1, 
        epochs=iterations, 
        seed=seed, 
        workers=workers)
    
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    
    return model
