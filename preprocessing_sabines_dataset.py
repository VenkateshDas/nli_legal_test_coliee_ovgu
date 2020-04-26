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

import xml.etree.ElementTree as ET                      #Needed to read the xml file of dataset.
from preprocessing import preprocessing as pre          #Calling the preprocessing.py 



file_path = "Sabine_training_data.xml"
xmlp = ET.XMLParser(encoding="utf-8")
tree = ET.parse(file_path, parser = xmlp)
root = tree.getroot()  

for pair in root.findall('pair'):
  PairID = pair.find('id').text 
  text1 = pair.find('t1').text 
  text2 = pair.find('t2').text 
  Label = pair.find('Label').text 
  
  changed_text1, has_Changes, change_type = pre.process(text1)   #Currently, I'm trying to track changes. Thus saving the changed text to another variable.
                                                    #Maybe in the real program we can replace the original text. 
  if has_Changes == True:
    change_description_string = "Text1 Change type: "+str(change_type)+"\t"+PairID 
    print(change_description_string)
    file_path = "sentence_change_log.txt"
    f = open(file_path, "a+")
    f.write(change_description_string+"\n"+text1+"\n")
    f.write(changed_text1)
    f.close()
  
  changed_text2, has_Changes, change_type = pre.process(text2)
  if has_Changes == True:
    change_description_string = "Text2 Change type:: "+str(change_type)+"\t"+PairID 
    print(change_description_string) 
    file_path = "sentence_change_log.txt"
    f = open(file_path, "a+")
    f.write(change_description_string+"\n"+text2+"\n")
    f.write(changed_text2)
    f.close()
  
