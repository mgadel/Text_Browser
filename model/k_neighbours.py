# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 17:30:05 2021

@author: mgadel
"""

from sklearn.neighbors import NearestNeighbors
from dataloader.data_tfidf import tf_idf
import numpy as np
import nltk

########################################
## K-NEIREST NEIGHBOURS ON DATA
########################################

'''
On cherche les documents les plus proches (k voisins) des mots clefs 
rentrés dans la QUERY.
suivant le compte TF IDF
'''

class model_knn:
    
    def __init__(self,data,n_neighbors=10):
        
        self.tfidf=data.tf_idf_corpus
        self.corpus=self.tfidf.corpus
        self.model = NearestNeighbors(n_neighbors=n_neighbors)
        self.model.fit(self.tfidf.tf_idf_matrix.toarray()) 
        
    
    def n_answer (self,query,col='Comment Text'):
        
        # need to tokennize as we trained the tf idf transformer on already tokkenized data
        qu=nltk.word_tokenize(query)
        query_vect = np.asarray(self.tfidf.tf_idf.transform([qu]).todense())
        
        neighbors_index = self.model.kneighbors(query_vect,return_distance=False)
        a=neighbors_index.tolist()
        answer = self.corpus[col].iloc[a[0]]
        
        return answer
    
    
'''
knn_model = model_knn(df_corpus,df_tf_idf)

knn_model.train_model(10)

'''