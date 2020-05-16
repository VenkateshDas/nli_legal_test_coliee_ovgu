"""
Developer:          Anirban Saha. 
Version:            v1.0 (released)
Date:               26.04.2020 
Description:        Contains functions that preprocesses the texts.
Documentation Link: [pending.]
Attachments:        bigram_.pkl 
                    legal_words.pkl 

Version History:
Version |   Date    |  Change ID |  Changes  

"""

"""
Contains functions related to preprocessing.
"""

from nltk import word_tokenize 
import pickle  
from spellchecker import SpellChecker
import numpy as np


"""
Description:    Downloads the bigram pkl file to local folder.
Input:           
Output:          
"""
def return_path_bigram():
    filename = "bigram_.pkl"
    filepath = "../../data/for_preprocessing/"+filename
    return filepath 


"""
Description:    Downloads the legal words pkl file to local folder.
Input:           
Output:          
"""
def return_path_legal_words():
    filename = "legal_words.pkl"
    filepath = "../../data/for_preprocessing/"+filename
    return filepath 


"""
Description:    Loading the Law2Vec word embedding model.  
"""
law2Vec_doc = "../../data/Law2Vec/Law2Vec.100d.txt" 
g_law2vec_wordmap = {}
with open(law2Vec_doc, "r", errors='ignore') as law2vec:
    for line in law2vec:
        name, vector = tuple(line.split(" ", 1))
        g_law2vec_wordmap[name] = np.fromstring(vector, sep=" ")


"""
Description:    Checks whether the word exists in Law2Vec 
Input:          word 
Output:         True: if the word exists. Else False. 
"""
def is_word_in_Dict(word):
  if word in g_law2vec_wordmap: 
    return True
  else:
    return False 
    
"""
Description:    Checks whether the word exists in the list of Legal Words. 
Input:          word 
Output:         True: if the word exists. Else False. 
"""
def is_word_in_legal_list(word, legal_words_list):
  if word in legal_words_list: 
    return True
  else:
    return False
    
"""
Description:    Returns a list of possible split words (into 2). 
Input:          word 
Output:         An array of possible split words.
"""
def return_sequence(word):  
  """
  Assumption: It will split into two words maximum.
  Why this assumption? In the dataset, all the examples where there are two words clubbed together,
  without a space in between can be split to a maximum of two words.
  """
  word_copy = word 
  word_list = []
  i = 1
  while i<=len(word):
    word1 = word[0:i]
    word2 = word_copy.replace(word1, '') 
    is_word_1 = is_word_in_Dict(word1)
    is_word_2 = is_word_in_Dict(word2)
    if is_word_1 ==1 and is_word_2 == 1:  
      word_list.append([word1, word2])
    i = i+1
  return word_list

"""
Description:    Returns index of bigram with highest frequency. 
Input:          array of bigrams. 
Output:         index of highest frequency among valid options. -1 if all options are invalid. 
"""
def get_replacement(array_, bigram):
  highest_score = 0
  highest_score_index = -1      #initialised. 
  index = 0
  replacement = ''
  
  for element in array_: 
    key = element[0] + " " + element[1] 
    if key in bigram: 
        if bigram[key] > highest_score:
            highest_score = bigram[key] 
            highest_score_index = index
    index = index + 1
    
  if highest_score_index == -1: # i.e. it could find nothing, then make exceptions.
  # Due to cultural differences, people from many cultures tend to modify English to suite their mother tongue.
  # For example: Germans are more comfortable saying "not relevant" instead of "irrelevant". 
  # For East Asians and Indians, they tend to use "something less" (eg: faultless) to suggest negation. This applies to mostly negations. 
    index = 0
    for element in array_:
        if element[1] in ["less", "not"]: 
            replacement = element[0] + " " + element[1]
        if element[1] in ["s"]: 
            replacement = element[0]
        index = index + 1 
  else: #i.e. it found a valid bigram to return. 
    replacement_arr = array_[highest_score_index]
    replacement = replacement_arr[0] + " " + replacement_arr[1]
  return replacement


"""
Description:    Takes a sentence as an input and should there be changes, it returns a changed sentence. 
Input:          sentence.
Output:         changed sentence. 
"""
def process(sentence): 
  
  # Declaring the variables, because why not?  
  has_Changes = False                   #Indicates if the text had changes. #This is not needed. 
                                        #It is done to manually check changes and if the changes fit the context. 
  change_type = 0                       #Indicates the type of change done. #This is not needed. 
  replacement_list = []                 #array where all changes would be consolidated.   
  to_log = ''                           #line in the output file.
  replacement = ''                      #replacement of the mistaken word. 
  spell = SpellChecker()

  # This is the list of primary changes that needs to be done to the sentence before any preprocessing.
  # It also contains changing words, whose English spellings are incorrect but heavily recurrent in the dataset. 
  # The expectation is, it would also occur in the future. 
  primary_changes = [
                     ["..", "."],
                     [".", " . "],
                     [",", " "],
                     [":"," "],
                     ["/", " / "],
                     ["n't", " not "],
                     ["cannot", "can not"], 
                     ["sublessee", "sub-lessee"],           #Both spellings found in the dataset, and both are the same word.  
                     ["renumeration", "remuneration"],      #Japanese cultural influence. 
                     ["supecify", "specify"],               #Japanese cultural influence.
                     ["superficy", "specify"]               #Japanese cultural influence.
                    ]
  
  for change in primary_changes:
    if change[0] in sentence:
       sentence = sentence.replace(change[0],change[1])
       has_Changes = True           #This is not required. But I'm currently manually checking all changes. 
       change_type = 1              #Primary changes. 
      
  # tokenising the sentence into words (probably), using NLTK punkt tokeniser. 
  tokens = word_tokenize(sentence.lower())
  
  # Load preexisting word sets
  # Description of each set would be given along side.
  
  # Set of Legal words which are not existing in Law2Vec. 
  # We do not plan to ignore this list of words, but initialise this separately in the main program. 
  legal_word_file_path = return_path_legal_words()
  with open(legal_word_file_path, 'rb') as file: 
      legal_words_list = pickle.load(file) 
      
  # Set of bigrams used in German law, Japanese law texts, Hamburg website () for legal discussion, Pennsylvania Code .
  # The purpose of this is to check the validity of bigrams generated by our program.
  # This can be bettered with more training. But currently we lack time and resources. 
  bigram_file = return_path_bigram()
  with open(bigram_file, 'rb') as file: 
      bigram = pickle.load(file)
  
  ## Set of words used in German law and Japanese law texts. 
  #word_file = "word_japanese_civil_code.pkl"
  #with open(word_file, 'rb') as file: 
  #    word_dict = pickle.load(file)  
  
  for t in tokens:
    sub_t = []
    
    #Check if t has hyphen.
    if '-' in t:
        copy_t = t.replace("-","")
        is_number = copy_t.isnumeric()          
        if is_number is False:  #then it is a character string. #if it is a number, sub_t is [] and the subsequent for-loop will not run. 
            is_word_law2Vec = is_word_in_Dict(t) 
            if is_word_law2Vec is False: # if it is true,i.e. if it is in Law2Vec, we do not need to process it. 
                sub_t = t.strip().split("-")            #sub_t might have 1 or more entries. 
                #print(sub_t) 
    else: 
        is_number = t.isnumeric()
        if is_number is False:  #it is not a number. If it is a number, we do not need to process it. 
                                #here we eliminate all possibilities of processing a number. 
            sub_t = [t.strip()] # in case there is no hyphen, then the word gets assigned to the array. It will have only single entry. 

    for word in sub_t: 
        word = word.strip()         # We remove space because while splitting, there are instances of a blank space.          
        if (word): #it should not be a blank space. 
            # Checks if the word is present in Law2Vec. 
            # We assume that any word present in Law2Vec is valid and without a mistake.  
            # If the word is there in the Legal word list, we ignore for now. It is manually made and is trusted to be valid and without a mistake. 
            is_word_law2Vec = is_word_in_Dict(word)
            is_word_legal = is_word_in_legal_list(word, legal_words_list)
            if is_word_law2Vec is False and is_word_legal is False:
                possible_replacements = return_sequence(word)
                
                #There might be more than one possible replacements. We need to choose a maximum of one from them.
                #We compare the options for their validity with the bigram index.
                #If more than one are valid, we choose the one which is most likely by the frequency count. 
                replacement = get_replacement(possible_replacements, bigram)
                
                if len(replacement) == 0:               # we got no replacement back from the function.
                    misspelled = spell.unknown([word])  # we use the spell correction package. 
                    if (misspelled):                    # if the spell correction package sends a replacement, 
                        replacement = spell.correction(misspelled.pop())   # we get the replacement word. 
                    else:
                        replacement = word              # in case every way of getting a replacement fails, we substitute the replacement with the word.  
                
                # Consolidates the changes in an array to change the main text before returning it to the calling function. 
                replacement_pair = []
                replacement_pair.append(word)
                replacement_pair.append(replacement)
                replacement_list.append(replacement_pair)
                
                # Makes a log of the changes, because why not? 
                to_log = word + "," + str(replacement)
                file_op = open("../../logs/word_change_log.csv","a+") 
                file_op.write(to_log + '\n')
                file_op.close()
                
                # Initialises the variables. 
                replacement = ''
                possible_replacements = [] 
  

      
  if (replacement_list): #If there are replacements. 
  #Changes to the main sentence before returning the sentence to the calling function. 
    for change in replacement_list:
      sentence = sentence.replace(change[0],change[1])
    has_Changes = True #Indicates to the calling function that there are changes. This is not required. I'm using it for manual testing purposes. 
    change_type = 2 #Word Replacements by Law2Vec, Splitting, SpellChecker etc. 
  else: 
    has_Changes = False #Indicates to the calling function that there are no changes. 
  
  return sentence
  
