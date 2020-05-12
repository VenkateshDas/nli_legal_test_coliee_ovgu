# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:58:23 2020

@author: Sachin Nandakumar
"""

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
import pandas as pd
from nltk import word_tokenize 
from nltk import pos_tag
from nltk.tag.mapping import tagset_mapping

'''
Location of file(s) required to run the program
'''

law2Vec_doc = "Law2Vec.200d.txt" 

'''
Define & initialize global constants
'''

word_dimension = 200
max_premise_length, max_hypothesis_length = 200, 80
PTB_UNIVERSAL_MAP = tagset_mapping('en-ptb', 'universal')
POS_categories = {'.', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X'}
tag_dimension = len(POS_categories)

# Read law2vec vectors from law2Vec_doc and store in a dictionary as [word:vector]
law2vec_wordmap = {}
with open(law2Vec_doc, "r", errors='ignore') as law2vec:
    for line in law2vec:
        name, vector = tuple(line.split(" ", 1))
        law2vec_wordmap[name] = np.fromstring(vector, sep=" ")
    del law2Vec_doc, line, name, vector                         # delete variables no longer required to free the RAM
        
        
def to_universal(tagged_words):
    return [(word, PTB_UNIVERSAL_MAP[tag]) for word, tag in tagged_words]

def get_POS_tags(text):
    pos_tagged = to_universal([(word, tag) for word, tag in pos_tag(word_tokenize(text))])
    pos_tagged = pd.DataFrame(pos_tagged, columns=['word', 'tag'])
    return pos_tagged


def fit_to_size(matrix, tag, shape): 
    '''
    Description:    Returns clipped-off/padded (premise/hypothesis) sentence
    Input:          1. premise/hypothesis sentence, 2. Desired output shape 
    Output:         Sentences either clipped-off at length limit or padded with law2vec embedding of word "PAD"
    '''
    padded_matrix = np.tile(law2vec_wordmap["PAD"],(shape[0],1))                    # initially create a matrix of desired shape with law2vec embeddings of word "PAD"
    padded_tags = np.zeros((shape[0], tag_dimension))
    slices = [slice(0,min(dim,shape[e])) for e, dim in enumerate(matrix.shape)]
    padded_matrix[slices] = matrix[slices]
    padded_tags[slices] = tag[slices]
    padded_matrix = np.concatenate((padded_matrix, padded_tags), axis=1)
    
    return padded_matrix


def sentence2sequence(sentence):
    '''
    Description:    Returns sequence of embeddings corresponding to each word in the sentence. Zero vector if OOV word 
    Input:          Sentence
    Output:         List of embeddings for each word in the sentence, List of words
    '''
    
    pos_tagged = get_POS_tags(sentence)
    vocabulary = list(pos_tagged[pos_tagged.columns[0]])
      
    rows, words = [], []
    for word in vocabulary:
        word = word.strip()
        if word in law2vec_wordmap:
            rows.append(law2vec_wordmap[word])
        else:
            rows.append(np.zeros((word_dimension)))
        words.append(word)    
        
    words_omitted_by_law2vec = list(set(vocabulary) - set(words))
    pos_tagged = pos_tagged[~pos_tagged['word'].isin(words_omitted_by_law2vec)].reset_index(drop=True)
    finalized_tags = pos_tagged[pos_tagged.columns[1]]
    #one hot encoding the pos tags
    finalized_tags = pd.get_dummies(finalized_tags)
    
    #find remaining categories of POS tags in POS_categories which were not present in the current sentence
    remaining_categories = list(POS_categories - set(list(finalized_tags)))
    
    #append those remaining categories to finalized_tags and sort them.
    if remaining_categories:
        empty_df = pd.DataFrame(0, index=np.arange(len(words)), columns=remaining_categories)
        finalized_tags = pd.concat([finalized_tags, empty_df], axis=1)
    finalized_tags = finalized_tags.reindex(sorted(finalized_tags.columns), axis=1)
    finalized_tags = finalized_tags.to_numpy()
    rows = np.vstack(rows)
    
    return (rows, finalized_tags)

    

def add_sentence_tags(premise, hyp, pad_dimension):
    '''
    Description:    Returns concatenated premise and hypothesis sentences
    Input:          Premise and hypothesis embedded sequences
    Output:         Numpy array of concatenated pairs of premise & hypothesis sequences
    '''
    
    # Reshaping vector of shape (1, word_dimension) to (1, 1, word_dimension)
    #   This means: 1 sentence consists of 1 word of 'word_dimension' dimensions
    # Tile operation: The reshaped vector is duplicated 627 times (total number of sentences): Now it becomes (627, 1, word_dimension)
    #   This means: 627 sentences consists of 1 word of 'word_dimension' dimensions
    BOS = np.tile(np.reshape(law2vec_wordmap["BOS"], (1, 1, pad_dimension)),(premise.shape[0],1,1))    # Vector for beginning of sentence
    SEP = np.tile(np.reshape(law2vec_wordmap["SEP"], (1, 1, pad_dimension)),(hyp.shape[0],1,1))        # Vector for separator between premise and hypo
    EOS = np.tile(np.reshape(law2vec_wordmap["EOS"], (1, 1, pad_dimension)),(hyp.shape[0],1,1))        # Vector for end of sentence

    BOS_premise = np.concatenate((BOS, premise), axis=1)                            # concat BOS vector to beginning of premise:- Concatenate to axis 1, since it is word-level concatenation 
    BOS_premise_SEP = np.concatenate((BOS_premise, SEP), axis=1)                    # concat SEP to end of premise 
    BOS_premise_SEP_hyp = np.concatenate((BOS_premise_SEP, hyp), axis=1)            # concat hypothesis to end of SEP
    BOS_premise_SEP_hyp_EOS = np.concatenate((BOS_premise_SEP_hyp, EOS), axis=1)    # concat EOS to end of hypothesis
    
    return BOS_premise_SEP_hyp_EOS



def get_data(preprocessed_json_file, datatype="TRAIN", model="baseline"):
    '''
    Description:    Returns embedded sequence of sentences which serves as input for LSTM network
    Input:          1. preprocessed_json_file: which consists of premise, hypothesis & labels (only for datatype=TRAIN)
                    2. datatype: Accepts two values: 
                        TRAIN - refers to training set. Preprocessed training set consists of 'labels' within the json file
                        TEST  - refers to test set. Preprocessed test set does not have 'labels' included within the json. It is in a separate txt file
    Output:         Output of add_sentence_tags() method, labels (only for datatype=TRAIN)
    '''
    
    global max_premise_length, max_hypothesis_length, word_dimension, POS_categories
    with open(preprocessed_json_file, 'r') as fp:   
        data = json.load(fp)
    
    premise_sentences = []
    hyp_sentences = []
    labels = [] 
    premise_tags = []
    hyp_tags = []
    
    for _, pair in data.items():
        premise = sentence2sequence(pair['text1'], model)          # pair['text1'] represents premise sentence
        premise_sentences.append(np.vstack(premise[0]))
        premise_tags.append(premise[1])
        
        hyp = sentence2sequence(pair['text2'], model)              # pair['text2'] represents hypothesis sentence
        hyp_sentences.append(np.vstack(hyp[0]))
        hyp_tags.append(hyp[1])
        
        if datatype == "TRAIN":
            labels.append(pair['label'])
    
    premise_sentences = np.stack([fit_to_size(sent, tag, (max_premise_length, word_dimension)) for sent, tag in zip(premise_sentences, premise_tags)])
    hyp_sentences = np.stack([fit_to_size(sent, tag, (max_hypothesis_length, word_dimension)) for sent, tag in zip(hyp_sentences, hyp_tags)])
    pad_dimension = word_dimension + tag_dimension
    
    if datatype == "TRAIN":
        return add_sentence_tags(premise_sentences, hyp_sentences, pad_dimension), labels
    else:
        return add_sentence_tags(premise_sentences, hyp_sentences, pad_dimension)
    
#get_data(train_set, max_premise_length, max_hypothesis_length)