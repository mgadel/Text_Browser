# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:56:49 2021

@author: mgadel
"""



import json

class configuration:
    """ config class which contains the raw data path, and info on model and model hyperparameter """
    
    def __init__(self, data_path, model_hyperparameter, cleaning_hyperparameter):        
        self.data_path = data_path
        self.model_hyperparameter = model_hyperparameter
        self.cleaning_hyperparameter = cleaning_hyperparameter        
        
    @classmethod 
    def load_json(cls,config_json):
        """ create a new instance of class config from config.json"""        
        with open(config_json,'r') as json_file:
            config = json.load( json_file)
            return cls(config['data']['path'], config['model'], config['cleaning'])