# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 18:09:49 2019

@author: isabe
"""
import warnings
import numpy as np
import pandas as pd
import os
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn.base import BaseEstimator
    from data_manager import DataManager # The class provided by binome 1
    # Note: if zDataManager is not ready, use the mother class DataManager

input_dir = "C:\\Users\\isabe\\Downloads\\monet-master\\starting_kit\\c1_input_data"

data_name = 'perso'
data_dir = '../../public_data'   
D = DataManager(data_name, data_dir, replace_missing=True)
#print(D)
print(data_name)
from data_io import read_as_df
data = read_as_df(data_dir  + '/' + data_name)          # The perso_data is loaded as a Pandas Data Frame
model_dir = '../sample_code_submission/'                        # Change the model to a better one once you have one!
result_dir = '../sample_result_submission/' 
problem_dir = '../ingestion_program/'  
score_dir = '../scoring_program/'
from sys import path; path.append(model_dir); path.append(problem_dir); path.append(score_dir); 

from numpy.core.umath_tests import inner1d
from data_io import write
from model import model

M = model()
trained_model_name = model_dir + data_name

if not(M.is_trained):
    X_train = D.data['X_train']
    Y_train = D.data['Y_train']
    # Fit the grid search to the data
    #grid_search.fit(X_train, Y_train)
    
    # affiche les meilleurs parametres 
    #print(grid_search.best_params_)
    M.fit(X_train, Y_train)                     

Y_hat_train = M.predict(D.data['X_train'])
Y_hat_valid = M.predict(D.data['X_valid'])
Y_hat_test = M.predict(D.data['X_test'])

print(Y_hat_train.shape)

M.save(trained_model_name)                 
result_name = result_dir + data_name
from data_io import write
write(result_name + '_train.predict', Y_hat_train)
write(result_name + '_valid.predict', Y_hat_valid)
write(result_name + '_test.predict', Y_hat_test)

from libscores import get_metric
metric_name, scoring_function = get_metric()
print('Using scoring metric:', metric_name)


from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
n_classes=2
def fpr_tpr(solution, prediction):
    for i in range(n_classes):
        fpr, tpr, _ = metrics.roc_curve(solution, prediction)
        roc_auc = metrics.auc(fpr, tpr)
    return (fpr,tpr)
def p2c(prediction,threshold=0.5) : 
    c = []
    for ele in prediction : 
        if(ele>=0.5) : 
            c.append(1)
        else : 
            c.append(0)
    return np.array(c)

def plot_cm_matrix(solution,prediction,title) :
    prediction = p2c(prediction)
    cm = confusion_matrix(solution, prediction)
    df_cm = pd.DataFrame(cm, index = [i for i in "01"],columns = [i for i in "01"])
    plt.figure(figsize = (5,3))
    sn.heatmap(df_cm, annot=True)
    plt.title(title)

def plot_ROC(fpr,tpr,title) :
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.show()
#if(os.path.exists("C:\\Users\\isabe\\Pictures\\monet-master\\monet-master\\public_data\\perso_train.solution")) : 
#    Y_test = D.data['Y_test']
#    Y_valid = D.data['Y_valid']
#
#if (os.path.exists("C:\\Users\\isabe\\Pictures\\monet-master\\monet-master\\public_data\\perso_train.solution")): 
#    fpr_train, tpr_train = fpr_tpr(Y_train, Y_hat_train)
##    fpr_test,tpr_test = fpr_tpr(Y_test, Y_hat_test)
##    fpr_valid,tpr_valid = fpr_tpr(Y_valid, Y_hat_valid)
##
#    print('Training score for the', metric_name, 'metric = %5.4f' % scoring_function(Y_train, Y_hat_train))
#    print('Ideal score for the', metric_name, 'metric = %5.4f' % scoring_function(Y_train, Y_train))
##    print('Test score for the', metric_name, 'metric = %5.4f' % scoring_function(Y_test, Y_hat_test))
##    print('Valid score for the', metric_name, 'metric = %5.4f' % scoring_function(Y_valid, Y_hat_valid))
##
#    plot_cm_matrix(Y_train,Y_hat_train,"Confusion matrix for train data") 
#    plot_ROC(fpr_train,tpr_train,"ROC curve for train data")
##    plot_cm_matrix(Y_test,Y_hat_test,"Confusion matrix for test data") 
##    plot_ROC(fpr_test,tpr_test,"ROC curve for test data")
##    plot_cm_matrix(Y_valid,Y_hat_valid,"Confusion matrix for valid data") 
##    plot_ROC(fpr_valid,tpr_valid,"ROC curve for valid data")
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
scores = cross_val_score(M, X_train, Y_train, cv=5, scoring=make_scorer(scoring_function))
print('\nCV score (95 perc. CI): %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))