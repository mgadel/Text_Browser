# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:14:19 2021

@author: mgadel
"""


##############################################################################
## PRE TRAINED METHOD + SMOOTH INVERSE FREQUENCY + COSINE SIMILART
##############################################################################


print('Define and Load Embedding Matrix')

# ATTENTION ne pas LEMMISER

vectorizer_emb = CountVectorizer(analyzer='word',stop_words=None,tokenizer=dummy,preprocessor=dummy)
data_vectorized = vectorizer_emb.fit_transform(DATA_eng['token_emb'])

vectorizer_emb.get_feature_names()


# Load GloVes Function

#define a max of value to work from the matrix embedding
max_features_embedding =25000
embed_size=300



# Load Dict and Embeddings

embedding_matrix,dict_emb,Glove_Index =Load_GloVes(EMBEDDING_FILE,vectorizer_emb.get_feature_names())     



'''
Smooth Inverse Frequency
Taking the average of the word embeddings in a sentence (as we did just above) tends to give 
too much weight to words that are quite irrelevant, semantically speaking. 
Smooth Inverse Frequency tries to solve this problem in two ways:
Weighting: SIF takes the weighted average of the word embeddings in the sentence. 
Every word embedding is weighted by a/(a + p(w)), where a is a parameter that is typically 
set to 0.001 and p(w) is the estimated frequency of the word in a reference corpus.
Common component removal: SIF computes the principal component of the resulting embeddings 
for a set of sentences. It then subtracts from these sentence embeddings their projections 
on their first principal component. This should remove variation related to frequency and 
syntax that is less relevant semantically
'''

def Smooth_Inv_Frequency():
    
    return



def Text_Emb(token_vect,dict_emb,Glove_Index):
    '''
    Return vector. Sum of embeddings of each words of a given text (vect = tokkenize text)
    '''
    length = len(dict_emb['true'])
    text_representation=np.zeros((length,))
    
    for word in token_vect:
        if word in Glove_Index:
            text_representation += dict_emb[word]

    
    return text_representation


def Corpus_Emb(corpus,dict_emb,Glove_Index):
    '''
    Return vector. Sum of embeddings of each words of a given text (vect tokkenize text)
    '''
    length_emb = len(dict_emb['true'])
    length_corpus = corpus.shape[0]
    
    corpus_emb = np.zeros((length_corpus,length_emb))
    
    for i in range(0,length_corpus):
        corpus_emb[i,:]=Text_Emb(corpus.iloc[i],dict_emb,Glove_Index)
        
    
    return corpus_emb


corpus_sum_emb = Corpus_Emb(DATA_eng['token_emb'],dict_emb,Glove_Index)

Vect_emb = Text_Emb(Q,dict_emb,Glove_Index)


def cosine_sim_texts(vector,corpus_sum_emb):
    '''
    Cosine similarity between vector and a corpus. Vector tokkenize vectoe. dict emb dictionnary
    with each embedding
    '''
        
    liste = []

    # transform vector to a 1D numpy array
    vect = np.asarray(vector).ravel()
    
    for i in range(0,corpus_sum_emb.shape[0]):
        
        vectEMB_i=np.asarray(corpus_sum_emb[i]).ravel()       
        score=cosine_sim(vectEMB_i, vect)
        liste.append((i,score) )      
        liste.sort(key = takeSecond,reverse=True)
               
    return liste

liste_emb = cosine_sim_texts(Vect_emb,corpus_sum_emb)


def n_answer(liste_emb,n=10,DATA_texte):
    #show N first 
    N_first=pd.DataFrame(columns=DATA_texte.columns)

    for i in range (0,N):
        print(N_first)
        key=liste_emb[i][0]
        N_first=N_first.append(DATA_texte.iloc[key], ignore_index = True)
        print(DATA_texte.iloc[key])

    return N_first

N_first = N_first_cosim(liste_emb,10,DATA_eng)




# INPUT QUERY AND TRANSFORM
'''
on transforme la question grace a TF_IDF
'''
QUERY='escape areas fore peak'
Q=nltk.word_tokenize(QUERY)

Vect_emb = Text_Emb(Q,dict_emb,Glove_Index)
liste_emb = cosine_sim_texts(Vect_emb,corpus_sum_emb)

N_first = N_first_cosim(liste_emb,10,DATA_eng)

