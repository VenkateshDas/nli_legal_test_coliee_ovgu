import json
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from data_parser import sentence2sequence

#TO DO : Document the code

preprocessed_json_file = '/Users/venkateshmurugadas/Documents/nli_coliee/nli_legal_test_coliee_ovgu_local/preprocessed_training_set.json'

with open(preprocessed_json_file, 'r') as fp:
    data = json.load(fp)

premise_sentences = []
hyp_sentences = []
premise = []
hypothesis = []

for _, pair in data.items():
    premise_sentences.append(pair['text1'])
    hyp_sentences.append(pair['text2'])

for i in range(len(premise_sentences)):

    p = premise_sentences[i]
    h = hyp_sentences[i]

    # p = word_tokenize(p)
    # h = word_tokenize(h)
    _,p = sentence2sequence(p)
    _,h = sentence2sequence(h)

    if len(p) > 200 :
        # print(len(p))
        # print(p)
        p = p[:200]

    if len(h) > 80 :
        # print(len(h))
        # print(h)
        h = p[:80]

    premise.append(p)
    hypothesis.append(h)

print(len(premise))
print(len(hypothesis))

new = []

for i in range(len(premise)):

    # print("Before premise" + str(len(premise[i])))

    if len(premise[i]) < 200:
        # r = 200 - len(premise[i])
        # for k in range(r):
        #     if len(premise[i]) < 200:
        #         premise[i].append('PAD')
        #     else:
        #         break
        while len(premise[i]) < 200:
            premise[i].append('PAD')

    # if len(premise[i]) > 200:
    #     print(premise[i])
    # print("After premise" + str(len(premise[i])))

    # print("Before hypothesis" + str(len(hypothesis[i])))
    if len(hypothesis[i]) < 80:
        # r = 80 - len(hypothesis[i])
        # for k in range(r):
        #     hypothesis[i].append('PAD')
        while len(hypothesis[i]) < 80:
            hypothesis[i].append('PAD')

    temp = 'BOS' + ' '
    temp += ' '.join(premise[i])
    temp += ' '
    temp += 'SEP'
    temp += ' '
    temp += ' '.join(hypothesis[i])
    temp += ' '
    temp += 'EOS'
    temp += '\n'

    new.append(temp)

print(f"Total strings : {len(new)}")

with open('output_text.pkl','wb') as t:
    pickle.dump(new,t)

f = open('output_text.txt','w')
for item in new:
    f.write(item)

sum = 0
count = 0
extra = 0
for item_ in new:
    text = item_
    text = text.split()
    if len(text) > 283:
        print(len(text))
        count += 1
        extra += len(text)
        # print(text)
    sum += len(text)

print(f"Total timesteps for 627 instances {str(sum)}")

print(f"Total instances with more than 283 words {str(count)}")
print(f"Total timesteprs with more than 283 words {str(extra)}")
