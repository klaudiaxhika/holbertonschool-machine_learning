#!/usr/bin/env python3
import math

def ngram_bleu(references, sentence, n):
    # Count n-grams in sentence
    sentence_ngrams = {}
    for i in range(len(sentence) - n + 1):
        ngram = tuple(sentence[i:i+n])
        sentence_ngrams[ngram] = sentence_ngrams.get(ngram, 0) + 1
    
    # Count n-grams in reference translations
    reference_ngrams = {}
    for reference in references:
        for i in range(len(reference) - n + 1):
            ngram = tuple(reference[i:i+n])
            reference_ngrams[ngram] = max(reference_ngrams.get(ngram, 0), reference.count(reference[i:i+n]))
    
    # Calculate precision for each n-gram
    total_precision = 0
    for ngram, count in sentence_ngrams.items():
        total_precision += min(count, reference_ngrams.get(ngram, 0))
    
    # Calculate BLEU score
    brevity_penalty = min(1.0, len(sentence) / max(sum(len(ref) for ref in references), 1))
    precision = total_precision / max(len(sentence), 1)
    bleu_score = brevity_penalty * (precision ** (1/n))
    
    return bleu_score

