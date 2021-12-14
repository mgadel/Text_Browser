# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:13:37 2021

@author: mgadel
"""

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import nltk

########################################
## COSINE SIMILARITY TO CHECK SIMILAR DOCUMENTS
########################################
'''
The tfidf represents your documents in a common vector space. If you then calculate
 the cosine similarity between these vectors, the cosine similarity compensates for 
 the effect of different documents' length. The reason is that the cosine similarity
 evaluates the orientation of the vectors and not their magnitude. I can show you 
 the point with python: Consider the following (dumb) documents
'''

def cosine_sim(a,b):
    try:
        cos_sim = np.dot(a, b)/(norm(a)*norm(b))
    
    except ValueError :
        cos_sim = 0
    
    except ZeroDivisionError:
        cos_sim=0
    
    return cos_sim 



# take second element for sort
def takeSecond(elem):
    return elem[1]



class model_cosine_sim:
    
    def __init__(self,data):
        
        self.tfidf=data.tf_idf_corpus
        self.corpus=self.tfidf.corpus
        
        

    def highest_score(self,query):   
        # Input : TF_IDF matrix
        # Query verctorized in TD IDF representation       
        liste = []  
        
        # transform vector to a 1D numpy array
        qu=nltk.word_tokenize(query)
        query_vect = np.asarray(self.tfidf.tf_idf.transform([qu]).todense())
        #query_tf_v = np.asarray(query_tf.todense()).ravel()
        
        # return the list of index with the highest cosine similarity score
        for i in range(0,self.tfidf.tf_idf_matrix.shape[0]):        
            vect_tf_idf_i=np.asarray(self.tfidf.tf_idf_matrix[i].todense())       
            score=cosine_sim(vect_tf_idf_i.ravel(), query_vect.ravel())
            liste.append((i,score))      
            liste.sort(key = takeSecond,reverse=True)               
        return liste


    def n_answer(self,query,n=10,col='Comment Text'):
        #show N first 
        N_first=[]
    
        cos_liste = self.highest_score(query)
    
        for i in range (0,n):
            key=cos_liste[i][0]
            N_first.append(self.corpus[col].iloc[key])
    
        return N_first






