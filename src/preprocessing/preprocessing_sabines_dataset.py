"""
Developer:          Anirban Saha. 
Version:            v1.0 (released)
Date:               26.04.2020 
Description:        Dummy Program to show how preprocessing.py is made to work.
Documentation Link: N.A. 
Attachments:        N.A. 

Version History:
Version |   Date    |  Change ID |  Changes  

"""

import xml.etree.ElementTree as ET          
import preprocessing as pre
import numpy as np
from tqdm import tqdm
import json

'''
GlOBAL CONSTANTS & FILES
'''
# File List:
RAW_TRAIN_DATA = "../../data/raw_data/Sabine_training_data.xml"

def fit_to_size(matrix, shape): 
    res = np.zeros(shape)
    slices = [slice(0,min(dim,shape[e])) for e, dim in enumerate(matrix.shape)]
    res[slices] = matrix[slices]
    return res

def get_data(data_file, data_type="TRAIN"):
    file_path = data_file
    train_json = dict()
    xmlp = ET.XMLParser(encoding="utf-8")
    tree = ET.parse(file_path, parser = xmlp)
    root = tree.getroot()  
    pbar = tqdm(total=len(root.findall('pair')))
    
    print('\nLoading premise and hypothesis sentences from disk...')
    print('-'*55)
    for id, pair in enumerate(root.findall('pair')):
#      pair_ID = pair.find('pair id').text 
      text1 = pair.find('t1').text 
      text2 = pair.find('t2').text 
      
      preprocessed_premise = pre.process(text1)
      preprocessed_hyp = pre.process(text2)
      
      temp = dict()
      temp['text1'] = preprocessed_premise.lstrip()
      temp['text2'] = preprocessed_hyp.lstrip()
      
      if data_type == "TRAIN":
          label = pair.find('Label').text 
          if label == 'Y': label = 1
          if label == 'N': label = 0
          temp['label'] = label
      
      train_json[id] = temp
      pbar.update(1)
      
    return train_json

if __name__ == "__main__":
    all_sentences = get_data(RAW_TRAIN_DATA)
        
    with open('../../data/preprocessed_data/preprocessed_training_set.json', 'w') as fp:
        json.dump(all_sentences, fp)    
        
    print('Preprocessing Complete!')
    
    print('File {} saved to {}'.format('preprocessed_training_set.json', '../../data/preprocessed_data/'))