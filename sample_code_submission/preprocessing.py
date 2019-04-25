
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 1 09:54:48 2019

@author: isabe
"""

"""
Il s'agit d'une deux fonctions de préprocessings, suivi de quelques tests pour s'assurer que nos données ont bien étés modifiés.
Dans le premier cas, un preprocessing avec PCA.
Le second cas, une selection des features les plus pertinants. 

Nous avons ensuite une fonction choixPrepro, qui nous sert a selectionner le preprocessing voulu, chose que nous
utiliserons dans les tests, qui sont fait dans le main.
"""
from sklearn.pipeline import Pipeline
import warnings
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.decomposition import PCA
from Visualisation import duTP
import pandas as pd
from sys import path; 


with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning) 
    from sklearn.base import BaseEstimator 
    from data_manager import DataManager # The class provided by binome 1 

    # Note: if zDataManager is not ready, use the mother class DataManager 


"""nombre de component pour PCA, et nombre de features pour selectKbest"""
nbcomponent = 0.98;
nbkfeatures = 190;
    
 
"""Il s'agit ici du preprocessing PCA"""
class Preprocessor(BaseEstimator):

    def __init__(self): 
        self.transformer =  PCA(n_components=nbcomponent)     

    def fit(self, X, y=None): 
        return self.transformer.fit(X, y) 


    def fit_transform(self, X, y=None): 
         return self.transformer.fit_transform(X) 


    def transform(self, X, y=None): 
        return self.transformer.transform(X) 
 
    """Notre deuxieme methode de preprocessing, le selectKbest"""
class Preprocessor2(BaseEstimator):

    def __init__(self): 
       self.transformer =  SelectKBest(f_regression, k=nbkfeatures) 

    def fit(self, X, y=None): 
        return self.transformer.fit(X, y) 

    def fit_transform(self, X, y=None): 
         return self.transformer.fit_transform(X, y) 
        
    def transform(self, X, y=None): 
        return self.transformer.transform(X) 

    
"""Une fonction pour appeler plus facilement les méthodes lors du test, ou nous combinons les methodes"""     
#Permet de tester les différents préprocesseurs
def choixPrepro(option):
    if option == 0: 
        print("\n\n*** Transformed data PCA ***")
        Prepro = Preprocessor() 
    elif option == 1: 
        print("\n\n*** Transformed SELECTKBEST ***")
        Prepro = Preprocessor2() 
    elif option == 2: 
        print("\n\n*** Transformed data PCA + SELECTKBEST ***")
        Prepro = Pipeline([('PCASelectKBest', Preprocessor()),('SelectKBest', Preprocessor2())])
    elif option == 3: 
        print("\n\n*** Transformed data SELECTKBEST + PCA ***")
        Prepro = Pipeline([('PCASelectKBest', Preprocessor2()),('SelectKBest', Preprocessor())])
    return Prepro 





"""C'est ici que nous testons le tout: nous checkons deja en sortie sur console, si les données varient,
  et sur leur "shape" entre les données de base et les differentes methode de preprocessing
  varient aussi comme on le souhaite. On double check en faisant appel à une methode du fichier visualisation.py
  qui permet de checker de manière plus visuel.
  Enfin, on triple check en les mettant en format csv pour voir s'ils sont bien formé en sortie.""" 
  
  
  
if __name__=="__main__":
# We can use this to run this file as a script and test the Preprocessor

    input_dir = "C:\\Users\\isabe\\Downloads\\monet-master\\starting_kit\\c1_input_data"
    output_dir = "./fichiers_preprocesses"
    basename = "perso"

    #Pour le test unitaire, on doit réduire le parametre k) 

    D = DataManager(basename, input_dir) # Load data 
    print("*** Original data ***") 
    print(D.data['X_train'].shape)
    print(D.data['X_train'])
    Ddf=  pd.DataFrame(D.data['X_train'], D.data['Y_train'])
    duTP(Ddf,True)
       

    for i in range(3): 
        Prepro = choixPrepro(i) 
        test = Prepro.fit_transform(D.data['X_train'], D.data['Y_train']) 
        # Here show something that proves that the preprocessing worked fine 
        print(test.shape) 
        print("Test ", i) 
        print(test)
        df = pd.DataFrame(test)
        duTP(df,True)
        nomfichier = 'test'+str(i)+'_train.data'
        df.to_csv(nomfichier, index=False, header=False)


