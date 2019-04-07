# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:54:48 2019

@author: isabe
"""

"""
Created on Sat Mar 11 08:04:23 2017
Last revised: Feb 2, 2019

@author: isabelleguyon

This is an example of program that preprocessed data.
It does nothing it just copies the input to the output.
Replace it with programs that:
    normalize data (for instance subtract the mean and divide by the standard deviation of each column)
    construct features (for instance add new columns with products of pairs of features)
    select features (see many methods in scikit-learn)
    re-combine features (PCA)
    remove outliers (examples far from the median or the mean; can only be done in training data)
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.decomposition import PCA

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn.base import BaseEstimator
    from data_manager import DataManager # The class provided by binome 1
    # Note: if zDataManager is not ready, use the mother class DataManager
   

class Preprocessor(BaseEstimator):
    def __init__(self):
        self.transformer =  PCA(n_components=0.98)
    
    def fit(self, X, y=None):
        return self.transformer.fit(X, y)

    def fit_transform(self, X, y=None):
        # self.transformer = SelectKBest(f_regression, k=100).fit_transform(X, y)
         #self.transformer = pd.DataFrame(self.transformer)
         return self.transformer.fit_transform(X)

    def transform(self, X, y=None):
        return self.transformer.transform(X)
 
class Preprocessor2(BaseEstimator):
    def __init__(self):
      
       self
        
    def fit(self, X, y=None):
        return self.transformer.fit(X, y)

    def fit_transform(self, X, y=None):
        # self.transformer =  PCA(n_components=0.98)
         #self.transformer = SelectKBest(f_regression, k=100).fit_transform(X, y)
        # bla = list(self.transformer)
       # bla  = self.transformer.astype(np.float64)
         #self.transformer = pd.DataFrame(self.transformer)
         self.transformer =  SelectKBest(f_regression, k=100).fit_transform(X, y)
         return self.transformer
     
    def transform(self, X, y=None):
        return self.transformer.transform(X)

    
    
if __name__=="__main__":
  # We can use this to run this file as a script and test the Preprocessor

    input_dir = "C:\\Users\\isabe\\Downloads\\monet-master\\starting_kit\\c1_input_data"
    output_dir = "./fichiers_preprocesses"
    
    basename = 'perso'

    D = DataManager(basename, input_dir) # Load data
    print("*** Original data ***")
    print(D)
    
    Prepro = Preprocessor()

    # Preprocess on the data and load it back into D
  #  D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
 #   D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
  #  D.data['X_test'] = Prepro.transform(D.data['X_test'])
    
    test = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])

    #D.feat_name = np.array(['PC1', 'PC2'])
   # D.feat_type = np.array(['Numeric', 'Numeric'])
  
    # Here show something that proves that the preprocessing worked fine
    print("*** Transformed data ***")
    print(test.shape)
   
   # D.data['X_train']  = SelectKBest(f_regression, k=100).fit_transform(D.data['X_train'], D.data['Y_train'])    
  
  #  print(test2)
  
   # print(D.data['X_train'].shape)
    test = pd.DataFrame(test)
    test.to_csv( 'PCA1_train.data', index=False, header=False)

