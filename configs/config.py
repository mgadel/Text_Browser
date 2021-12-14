# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 17:38:13 2021

@author: mgadel
"""


CONFIG = {
        "data": {
                "path": r"C:\Users\mgadel\Desktop\BV_IA_Perso\M&O Data Strategy\CODE_MGL\Question_Answer\data\raw"
                },
        "cleaning": {},
        "model":{
                'embeddings':{
                       'file': r"C:\Users\mgadel\Desktop\BV_IA_Perso\M&O Data Strategy\CODE_MGL\Question_Answer\data\external\glove.840B.300d\glove.840B.300d.txt",
                       "max_features_embedding":25000,
                       "embed_size" :300},
                'knn':{
                        "n_voisins":10
                        }
                }
            }
                
        
"""
import json

WRITE
with open('data.json', 'w') as outfile:
    json.dump(CONFIG, outfile)

READ
with open('config.json', 'r') as outfile:
    data = json.load(outfile)
            
    
config_glove={'file': r"C:\Users\mgadel\Desktop\BV_IA_Perso\M&O Data Strategy\CODE_MGL\Question_Answer\data\external\glove.840B.300d\glove.840B.300d.txt","max_features_embedding":25000,"embed_size" :300}

"""