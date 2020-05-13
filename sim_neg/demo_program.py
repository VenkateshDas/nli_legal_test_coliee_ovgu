from sim_neg import similarity_negation_vector as sn
import xml.etree.ElementTree as ET

"""
Description:    This is the main program. The name would change however. I'll comment on this later. 
Input:          Negation word. 
Output:         Prints a vector. So this would change as well. 
"""

negation_word = 'not'
ngram = 4                                                         #the team needs to decide this. 
file_path = 'Sabine_training_data.xml'
xmlp = ET.XMLParser(encoding="utf-8")
tree = ET.parse(file_path, parser = xmlp)
root = tree.getroot()

for pair in root.findall('pair'):
  PairID = pair.find('id').text 
  text1 = pair.find('t1').text 
  text2 = pair.find('t2').text 
  Label = pair.find('Label').text
  
  vector = sn.get_sim_vector_for_pair(text1, text2, ngram, negation_word)
  print(vector)
