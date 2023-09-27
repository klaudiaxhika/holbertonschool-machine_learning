#!/usr/bin/env python3
import math

import math

def ngram_precision(candidate_ngrams, reference_ngrams):
    count = sum(min(candidate_ngrams.get(ngram, 0), reference_ngrams.get(ngram, 0)) for ngram in candidate_ngrams)
    total = sum(candidate_ngrams.values())
    return count / total if total != 0 else 0

def cumulative_bleu(references, sentence, n):
    precisions = []
    
    for i in range(1, n + 1):
        candidate_ngrams = {}
        reference_ngrams = {}
        
        for reference in references:
            for j in range(len(reference) - i + 1):
                ngram = tuple(reference[j:j+i])
                reference_ngrams[ngram] = max(reference_ngrams.get(ngram, 0), reference.count(ngram))
        
        for j in range(len(sentence) - i + 1):
            ngram = tuple(sentence[j:j+i])
            candidate_ngrams[ngram] = candidate_ngrams.get(ngram, 0) + 1
        
        precision = ngram_precision(candidate_ngrams, reference_ngrams)
        precisions.append(precision)
    
    non_zero_precisions = [p if p != 0 else 1e-10 for p in precisions]
    brevity_penalty = min(1.0, len(sentence) / max(sum(len(ref) for ref in references), 1))
    geometric_mean = math.exp(sum(map(math.log, non_zero_precisions)) / n)
    cumulative_bleu_score = brevity_penalty * geometric_mean
    
    return cumulative_bleu_score

