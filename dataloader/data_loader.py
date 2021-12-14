# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:30:17 2021

@author: mgadel
"""

import pandas as pd
import os
import re
from dataloader.data_cleaning import df_clean_token_save
from dataloader.data_tfidf import tf_idf


class dataloader:
    
    def __init__(self,path):
        
        self._df_corpus_raw =self.load_data(path)
        self.df_corpus_clean=df_clean_token_save(self._df_corpus_raw,path,col = 'Comment Text') 
        self.tf_idf_corpus= tf_idf(self.df_corpus_clean)
    
    
    def load_data(self,path):     
        
        pattern = "\.xlsx"
        
        for i in os.listdir(path):
            if re.search(pattern,i) :
                return pd.read_excel(path+'\\'+ i)  







"""
class DataLoader:
    
    @staticmethod
    def load_data(path):     
        
        pattern = "\.xlsx"
        
        for i in os.listdir(path):
            if re.search(pattern,i) :
                return pd.read_excel(path+'\\'+ i)  
"""

