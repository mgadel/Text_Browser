# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:40:50 2021

@author: mgadel
"""


from sklearn.feature_extraction.text import TfidfVectorizer



# 4 - BAG OF WORDS METHOD / VECTORIZATION
''' It is basic model used in natural language processing. 
Why it is called bag of words because any order of the words in the document 
is discarded it only tells us weather word is present in the document or not 

So how a word can be converted to vector can be understood by simple word count 
example where we count occurrence of word in a document 

The approach which is discussed above is unigram because we are considering only 
one word at a time . Similarly we have bigram(using two words at a time- for example
 — There used, Used to, to be, be Stone, Stone age), trigram(using three words at a time
 - for example- there used to, used to be ,to be Stone,be Stone Age), ngram(using n words
 at a time)
Hence the process of converting text into vector is called vectorization.
By using CountVectorizer function we can convert text document to matrix of word count.
 Matrix which is produced here is sparse matrix. By using CountVectorizer on above document
 we get 5*15 sparse matrix of type numpy.int64.
 
TF-IDF stands for Term Frequency-Inverse Document Frequency which basically tells 
importance of the word in the corpus or dataset. TF-IDF contain two concept 
Term Frequency(TF) and Inverse Document Frequency(IDF)

Term Frequency
Term Frequency is defined as how frequently the word appear in the document or corpus. 
As each sentence is not the same length so it may be possible a word appears in long 
sentence occur more time as compared to word appear in sorter sentence. Term frequency 
can be defined as:

Inverse Document Frequency
Inverse Document frequency is another concept which is used for finding out importance 
of the word. It is based on the fact that less frequent words are more informative and 
important. IDF is represented by formula:

TF-IDF
TF-IDF is basically a multiplication between Table 2 (TF table) and Table 3(IDF table) .
 It basically reduces values of common word that are used in different document. 
 As we can see that in Table 4 most important word after multiplication of TF and IDF is 
 ‘TFIDF’ while most frequent word such as ‘The’ and ‘is’ are not that important
'''


def dummy(tokens):
    return tokens



class tf_idf:
    
    def __init__(self,corpus,col="Token"):
        self.corpus = corpus
        self.tf_idf = TfidfVectorizer(analyzer='word',tokenizer=dummy,token_pattern=None,lowercase=False)
        self.tf_idf_matrix = self.tf_idf.fit_transform(corpus[col].tolist())
        self.feature_names= self.tf_idf.get_feature_names()
        
    def tf_idf_vectorize(self,text):        
        return self.tf_idf.transform([text])
    




