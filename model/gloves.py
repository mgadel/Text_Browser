# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 17:31:54 2021

@author: mgadel
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import nltk

#config_glove={file = r"C:\Users\mgadel\Desktop\BV_IA_Perso\M&O Data Strategy\CODE_MGL\Question_Answer\data\external\glove.840B.300d\glove.840B.300d.txt"
#"max_features_embedding":25000
#"embed_size" :300}

def dummy(x):
    return x

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



class model_glove:

    def __init__(self, data,config_glove,col='Token'):
       
        # load glove config parameter
        self.file = config_glove['file']
        self.max_features_embedding = config_glove['max_features_embedding']
        self.embed_size=config_glove['embed_size']
        
        #load training data
        self.corpus = data.df_corpus_clean
        self.embedding_maxsize_corpus=max(data.df_corpus_clean[col])
        
        #transform training data as input for Glove
        self.counter = CountVectorizer(analyzer='word',stop_words=None,tokenizer=dummy,preprocessor=dummy)
        self.counter.fit(data.df_corpus_clean[col])
        self.feature_name = self.counter.get_feature_names()

        #load Glove and embedding corpus
        self.dataloader_glove()
        self.corpus_emb = self.corpus_emb()
    
    
    
    def dataloader_glove(self):
        '''
        Load the file of GloVe Matrix embeddings in the text file
        limit the size of the matrix to the worlds found in the training corpus (world_index)
        '''
        
        # Load all worlds in the GLOVE file
        with open(self.file,'r',encoding='utf8') as f:
            words= set()
            glove_dict={}
            
            for lines in f:
                tmpLine = lines.split(" ")
                curr_word=tmpLine[0]
                words.add(curr_word)
                glove_dict[curr_word] = np.array(tmpLine[1:], dtype='float32')
        
        #on prend le mean et la std
        #all_embs = np.stack(embeddings_index.values())
        words=list(glove_dict.values())
        emb_mean,emb_std = np.array(words).mean(), np.array(words).std()
        
        #on reduit la taille de la matrix embedding pour diminuer le temps de calcu
        nb_words_emb=min(self.max_features_embedding,len(self.feature_name))
        
        #On initialize la matrice avec la moyenne de tout // we add +1 to fit keras requirements
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words_emb, self.embed_size))
          
        # we build the dict_emb of all the world in the present text corpus
        i=0
        dict_emb={}

        
        for word in self.feature_name:
            if i>= self.max_features_embedding: 
                continue
            
            embedding_vector = glove_dict.get(word)
            # on traite le cas ou le mot ne soit pas dans le dico
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                
            dict_emb[word]= embedding_matrix[i]
                
            i+=i
            
        self.embedding_matrix = embedding_matrix
        self.dict_emb = dict_emb
        #self.glove_index = glove_index
            
        return 

   
    
    def text_emb(self,vect,already_tokkenized=True):
        '''
        Return avector wich is the sum of the embeddings of each words of a given text
        input = count vector
        output = size emb x 1
        '''
        length_emb = len(self.dict_emb[list(self.dict_emb.keys())[1]])
        text_representation=np.zeros((length_emb,))
        token_vect=vect 
        
        if already_tokkenized ==False :            
            token_vect = nltk.word_tokenize(vect)
        
        for word in token_vect:
            if (word in self.dict_emb.keys()) == True :
                text_representation = text_representation + self.dict_emb[word] 
        
        return text_representation
    
    
    
    def corpus_emb(self,col='Token'):
        '''
        Return the matrix corresponding to the text embedding of all corpus text (vect tokkenize text)
        output = size corpus x size_emb
        
        '''
        length_emb = len(self.dict_emb[list(self.dict_emb.keys())[1]])
        length_corpus = self.corpus.shape[0]
        
        corpus_emb = np.zeros((length_corpus,length_emb))
        
        for i in range(0,length_corpus):
            corpus_emb[i,:]=self.text_emb(self.corpus[col].iloc[i])
        
        return corpus_emb





    def cosine_sim_text(self,vect):
        '''
        Cosine similarity between raw vectorthe transformed corpus. Vector tokkenize vectoe. dict emb dictionnary
        with each embedding
        '''
            
        liste = []
    
        # transform  to a 1D numpy array
        vect = self.text_emb(vect, False)
        
        for i in range(0,self.corpus_emb.shape[0]):
            
            vect_emb_i=np.asarray(self.corpus_emb[i]).ravel()       
            score=cosine_sim(vect_emb_i, vect)
            liste.append((i,score) )      
            liste.sort(key = takeSecond,reverse=True)
                   
        return liste  
   
 
    
    def n_answer(self,text,n=10,col='Comment Text'):
        '''
        n closest answer to the corpus
        '''
        n_first=[]
    
        liste_answer = self.cosine_sim_text(text)
    
        for i in range (0,n):
            key=liste_answer[i][0]
            n_first.append(self.corpus[col].iloc[key])
    
        return n_first
    
