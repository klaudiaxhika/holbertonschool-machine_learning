#!/usr/bin/env python3

import tensorflow_hub as hub
from transformers import BertTokenizer

def question_answer(question, reference):
    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    
    # Load the BERT model from TensorFlow Hub
    model = hub.load("https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/4")
    
    # Tokenize the question and reference text
    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)
    
    # Combine question and reference tokens with [SEP] token in between
    input_tokens = ["[CLS]"] + question_tokens + ["[SEP]"] + reference_tokens + ["[SEP]"]
    
    # Convert tokens to token IDs
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    
    # Get answer span using BERT model
    inputs = {
        "input_word_ids": tf.constant([input_ids]),
        "input_type_ids": tf.constant([0] * len(question_tokens) + [1] * len(reference_tokens) + [1]),
        "input_mask": tf.constant([1] * len(input_tokens))
    }
    outputs = model(inputs)
    start_logits = outputs["start_logits"].numpy()[0]
    end_logits = outputs["end_logits"].numpy()[0]
    
    # Find the start and end indices of the answer span
    start_idx = np.argmax(start_logits)
    end_idx = np.argmax(end_logits[start_idx:]) + start_idx
    
    # Get the answer span from the reference text
    answer_tokens = input_tokens[start_idx:end_idx + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    
    return answer
