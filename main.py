# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 16:18:43 2021

@author: mgadel
"""

#from configs.config import CONFIG
from utils.load_config import configuration
from dataloader.data_loader import dataloader
from model.k_neighbours import model_knn
from model.cosine_similarity import model_cosine_sim
from model.gloves import model_glove
from sys import exit



PATH = r'configs\config.json'



def init():
    
    global config, data
    
    # Set Configuration Parameters
    config = configuration.load_json(PATH)
    # Load Data corpus
    data = dataloader(config.data_path)



def build_model():
        
    global model
   
    while True :
        model_choice =input('Select Model : knn, cosine sim, glove (to quit press Q): \n')
   
        if model_choice == 'knn':  
            model = model_knn(data,config.model_hyperparameter['knn']['n_voisins'])
            break
        
        elif model_choice == 'cosine sim':
            model = model_cosine_sim(data)
            break

        elif model_choice == 'glove':
            model = model_glove(data,config.model_hyperparameter['embeddings'])
            break
        
        elif model_choice == 'Q':
            print('\n...............\n')
            menu()
        
        else :
            print('Selection error')
            model_choice =input('Select Model : knn, cosine sim, glove (to quit press Q): \n')



def ask_questions():
    
    query = input('Enter keyworlds: \n')   
    print('\n')
    answers = model.n_answer(query)  
    other_ans = 'yes'
        
    for i, a in enumerate(answers):
        if i<3:
            print('Answer' + str(i+1) + '\n')
            print(a)
            print('\n')
                
        if i==3:
            other_ans = input('Print all answers (yes/no) ? \n')
                   
            while True:                
                if other_ans == 'no' or other_ans == 'yes' :
                    break
                else :
                    print('Wrong input ! Try again \n')     
                    other_ans = input('Print all answers (yes/no) ? \n')
            
        elif i>= 3 and other_ans == 'no':
            break
        
        elif i>= 3 and other_ans == 'yes':
            print('Answer' + str(i+1) + '\n')
            print(a)
            print('\n')

    

def menu():
    
    if 'model' not in globals():
        print('First, you should select a model to start searching ! \n')
        build_model()
    

    while True :
        
        print('\n --- Menu ---')
        menu_selection = input('New Model (Press M), New Request (Press R), Exit (Press Q): \n')

        if menu_selection == 'M':
            build_model()
            
        elif menu_selection == 'R':
            ask_questions()
            
        elif menu_selection == 'Q':
            ans =input('Are you sure to quit  (yes/no): \n')
             
            if ans =='no':
                menu()
                
            elif ans =='yes':
                break
        
        else :
            print('Selection error !')
            menu_selection =input('New Model (Press M), New Request (Press Q), Exit (Press Q): \n')
            
    print('\n Good Bye ! \n')
    exit()



def run_main():
    
    print('\n')
    print('Loading')
    print('...............\n')
    
    init()
        
    
    menu()





if __name__=='__main__':

    print('\n --- TEXT SEARCH --- \n')
    run_main()
    
    
    
