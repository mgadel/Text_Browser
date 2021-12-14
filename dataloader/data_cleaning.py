# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 17:24:48 2021

@author: mgadel
"""

import nltk
from nltk.corpus import stopwords
import re
from string import punctuation
from nltk.stem import WordNetLemmatizer
import numpy as np

# 1 - CLEANING THE DATA 
'''
In regular sentences Noisy data can be defined as text file header,footer, 
HTML,XML,markup data.As these type of data are not meaningful and does not provide 
any information so it is mandatory to remove these type of noisy data. 
In python HTML,XML can be removed by BeautifulSoup library while markup,
header can be removed by using regular expression
    
    TOKKENIZATION
    In tokenization we convert group of sentence into token . It is also called text 
    segmntation or lexical analysis. It is basically splitting data into small chunk of words.
    For example- We have sentence — “Ross 128 is earth like planet.Can we survive in that 
    planet?”. After tokenization this sentence will become -[‘Ross’, ‘128’, ‘is’, ‘earth’,
                                                             ‘like’, ‘planet’, ‘.’, ‘Can’, ‘we’, ‘survive’, ‘in’, ‘that’, ‘planet’, ‘?’]. Tokenization 
                                                             in python can be done by python’s NLTK library’s word_tokenize() function

    TOKKENIZER
    This tokenizer performs the following steps:
        - split standard contractions, e.g. don't -> do n't and they'll -> they 'll
        - treat most punctuation characters as separate tokens
        - split off commas and single quotes, when followed by whitespace
        - separate periods that appear at the end of line

# 3 - NORMALISATION // STEMMING - LEMMATIZATION

Before going to normalization first closely observe output of tokenization. 
Will tokenization output can be considered as final output? Can we extract 
more meanigful information from tokenize data ?
In tokenaization we came across various words such as punctuation,stop words
(is,in,that,can etc),upper case words and lower case words.After tokenization 
we are not focused on text level but on word level. So by doing stemming,lemmatization 
we can convert tokenize word to more meaningful words . For example — [‘‘ross’, ‘128’, 
‘earth’, ‘like’, ‘planet’ , ‘survive’, ‘planet’]. As we can see that all the punctuation 
and stop word is removed which makes data more meaningful

STEMMING = 'algorithms work by cutting off the end or the beginning of the word, 
taking into account a list of common prefixes and suffixes that can be found in an 
inflected word. This indiscriminate cutting can be successful in some occasions, but 
not always, and that is why we affirm that this approach presents some limitations.'
LEMMATIZATION = n the other hand, takes into consideration the morphological analysis of the 
words. To do so, it is necessary to have detailed dictionaries which the algorithm can 
look through to link the form back to its lemma. Again, you can see how it works with the
 same example words.
'''



puncts = [',', '.','°',"''", '"', ':', ')', '§','(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
'·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
'“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
'▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
'∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', '``','º', 'ºc']



def isNaN(x):
    return x!= x



def nltk_language(text):
    '''
    define a function which turn language into its language
    in output, the language with the higest reference count
    '''
    token = nltk.word_tokenize(text)
  
    dico={}
    
    for language in stopwords.fileids(): 
        stopwords_set=set(stopwords.words(language))
        text_set = set(token)
        intersect = text_set.intersection(stopwords_set)
        
        dico[language]= len(intersect)     
                
    max_occurence=max(zip(dico.values(), dico.keys())) 
    
    return max_occurence[1]



def select_english(x):
    if nltk_language(x)=='english':
        return x
    else :
        return 0
    


def clean_text(x):
    '''
    Remove commun punctuation
    All numbers
    remove small worlds (lower than 3chara)
    return low case
    '''
    x = str(x)
    
    for punct in list(puncts):
        x = x.replace(punct, f' {punct} ')
        
    x = re.sub('[0-9]+', '', x)   
    #remove smaller than 2 characters
    x = re.sub(r"\b[a-zA-Z]{,3}\b", '', x)
    
    return x.lower()



def tokkenize(x):
    return nltk.word_tokenize(x) if x!=0 else x



def remove_stopwords_and_punctuation(x,language='english'):
    stop_words_puncts = stopwords.words('english') + puncts

    if x !=0:
        x = [word for word in x if word not in stop_words_puncts]
    return x



"""
def lemming(x):
    wnl=WordNetLemmatizer
    print(x)
    return [ wnl.lemmatize(word) for word in x]    
 """  



def clean_sentence_and_tokkenize(sentence):
    
    #functions to be applied to sentence
    funct_list = [select_english, clean_text, tokkenize,remove_stopwords_and_punctuation]
    
    for i, fct in enumerate(funct_list):
        if (sentence !=0) and (isNaN(sentence) == False) :
            sentence = fct(sentence)         

    return sentence



def save_processed(df,path):
    '''
    Save intermediate processed data into /processed for debug
    '''
    path_processed = re.sub(r'raw',r'processed\\',path)
    df.to_csv(path_processed + 'data_processed_and_tokken.csv', index=False)   
    return
 
    

def df_clean_token_save(df,path,col = 'Comment Text'):
    '''
    tokkenize and save the intermediate data
    '''
    df['Token']=df[col].apply(clean_sentence_and_tokkenize)
    
    df=df.dropna()
    df=df[df['Token']!=0]
    
    save_processed(df,path)
    return df


