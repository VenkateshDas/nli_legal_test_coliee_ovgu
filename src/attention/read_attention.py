import pickle
import numpy as np
from text_attention import *
from nltk.tokenize import word_tokenize


file = "scores/attention/attention_scores.pkl"
text_file = "test.pkl"

f = open(file,'rb')
attention = pickle.load(f)
attention = np.vstack(attention)


g = open(text_file,'rb')
text_list = pickle.load(g)
print(len(text_list))

path = 'attention_visuals/'
file_name = 'sentence'

for i in range(len(text_list)):
    text = word_tokenize(text_list[i])
    print(i)
    print(len(text))
    generate(text , attention[i],path+file_name+str(i)+".tex")
