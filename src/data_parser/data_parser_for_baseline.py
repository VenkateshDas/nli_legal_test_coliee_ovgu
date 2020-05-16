# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:21:23 2020

@author: Sachin Nandakumar
"""

'''#######################################################
        PARSE PREPROCESSED TRAIN/TEST DATA
#######################################################'''


import json
import numpy as np
from nltk import word_tokenize 

'''
Location of file(s) required to run the program
'''

law2Vec_doc = "../data/Law2Vec/Law2Vec.200d.txt" 

'''
Define & initialize global constants
'''

word_dimension = 200
max_premise_length, max_hypothesis_length = 200, 80

# Read law2vec vectors from law2Vec_doc and store in a dictionary as [word:vector]
law2vec_wordmap = {}
with open(law2Vec_doc, "r", errors='ignore') as law2vec:
    for line in law2vec:
        name, vector = tuple(line.split(" ", 1))
        law2vec_wordmap[name] = np.fromstring(vector, sep=" ")
    del law2Vec_doc, line, name, vector                         # delete variables no longer required to free the RAM
        
        

def fit_to_size(matrix, shape): 
    '''
    Description:    Returns clipped-off/padded (premise/hypothesis) sentence
    Input:          1. premise/hypothesis sentence, 2. Desired output shape 
    Output:         Sentences either clipped-off at length limit or padded with law2vec embedding of word "PAD"
    '''
    padded_matrix = np.tile(law2vec_wordmap["PAD"],(shape[0],1))                    # initially create a matrix of desired shape with law2vec embeddings of word "PAD"
    slices = [slice(0,min(dim,shape[e])) for e, dim in enumerate(matrix.shape)]
    padded_matrix[slices] = matrix[slices]
    
    return padded_matrix


def sentence2sequence(sentence):
    '''
    Description:    Returns sequence of embeddings corresponding to each word in the sentence. Zero vector if OOV word 
    Input:          Sentence
    Output:         List of embeddings for each word in the sentence, List of words
    '''
    vocabulary = word_tokenize(sentence.lower())
    rows, words = [], []
    for word in vocabulary:
        word = word.strip()
        if word in law2vec_wordmap:
            rows.append(law2vec_wordmap[word])
        else:
            rows.append(np.zeros((word_dimension)))
        words.append(word)
    return rows, words

def add_sentence_tags(premise, hyp):
    '''
    Description:    Returns concatenated premise and hypothesis sentences
    Input:          Premise and hypothesis embedded sequences
    Output:         Numpy array of concatenated pairs of premise & hypothesis sequences
    '''
    
    # Reshaping vector of shape (1, word_dimension) to (1, 1, word_dimension)
    #   This means: 1 sentence consists of 1 word of 'word_dimension' dimensions
    # Tile operation: The reshaped vector is duplicated 627 times (total number of sentences): Now it becomes (627, 1, word_dimension)
    #   This means: 627 sentences consists of 1 word of 'word_dimension' dimensions
    BOS = np.tile(np.reshape(law2vec_wordmap["BOS"], (1, 1, word_dimension)),(premise.shape[0],1,1))    # Vector for beginning of sentence
    SEP = np.tile(np.reshape(law2vec_wordmap["SEP"], (1, 1, word_dimension)),(hyp.shape[0],1,1))        # Vector for separator between premise and hypo
    EOS = np.tile(np.reshape(law2vec_wordmap["EOS"], (1, 1, word_dimension)),(hyp.shape[0],1,1))        # Vector for end of sentence

    BOS_premise = np.concatenate((BOS, premise), axis=1)                            # concat BOS vector to beginning of premise:- Concatenate to axis 1, since it is word-level concatenation 
    BOS_premise_SEP = np.concatenate((BOS_premise, SEP), axis=1)                    # concat SEP to end of premise 
    BOS_premise_SEP_hyp = np.concatenate((BOS_premise_SEP, hyp), axis=1)            # concat hypothesis to end of SEP
    BOS_premise_SEP_hyp_EOS = np.concatenate((BOS_premise_SEP_hyp, EOS), axis=1)    # concat EOS to end of hypothesis
    
    return BOS_premise_SEP_hyp_EOS

def get_data(preprocessed_json_file, datatype="TRAIN"):
    '''
    Description:    Returns embedded sequence of sentences which serves as input for LSTM network
    Input:          1. preprocessed_json_file: which consists of premise, hypothesis & labels (only for datatype=TRAIN)
                    2. datatype: Accepts two values: 
                        TRAIN - refers to training set. Preprocessed training set consists of 'labels' within the json file
                        TEST  - refers to test set. Preprocessed test set does not have 'labels' included within the json. It is in a separate txt file
    Output:         Output of add_sentence_tags() method, labels (only for datatype=TRAIN)
    
    '''
    
    global max_premise_length, max_hypothesis_length
    with open(preprocessed_json_file, 'r') as fp:   
        data = json.load(fp)
    
    premise_sentences = []
    hyp_sentences = []
    labels = [] 
    
    premise_text = []
    hyp_text = []
    
    for _, pair in data.items():
        premise = sentence2sequence(pair['text1'])          # pair['text1'] represents premise sentence
        premise_sentences.append(np.vstack(premise[0]))
        premise_text.append(premise[1])
        
        hyp = sentence2sequence(pair['text2'])              # pair['text2'] represents hypothesis sentence
        hyp_sentences.append(np.vstack(hyp[0]))
        hyp_text.append(hyp[1])
        
        if datatype == "TRAIN":
            labels.append(pair['label'])
    
    
    premise_sentences = np.stack([fit_to_size(x, (max_premise_length, word_dimension))
                      for x in premise_sentences])
    
    hyp_sentences = np.stack([fit_to_size(x, (max_hypothesis_length, word_dimension))
                  for x  in hyp_sentences])
    
    if datatype == "TRAIN":
        return add_sentence_tags(premise_sentences, hyp_sentences), labels
    else:
        return add_sentence_tags(premise_sentences, hyp_sentences)
    
#get_data(train_set, max_premise_length, max_hypothesis_length)