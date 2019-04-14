# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 12:30:02 2019

@author: isabe
"""

#Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import seaborn as sns
import data_io

'''Il s'agit d'un fichier permettant la generation d'images'''


def funcPCA (data) : 

    print(np.cumsum(pca.explained_variance_ratio_))
    plt.figure(figsize = (8,8))
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('PCA Analysis')
    plt.ylim(30,100.5)
    plt.plot(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100))
    plt.show()
    plt.close()


def funcDiagbaton (donnee) :
    plt.figure(figsize = (15,10))
    
    


    if( donnee==1):
        plt.ylim(60,105)
        plt.title('Comparaison du score de Cross validation')
        fake = pd.DataFrame({'Type of classification': ['Optimal \nRandomForest\n without PCA', 'Optimal\n RandomForest\n with PCA', 'Default \nRandomForest\nwithout PCA', 'Optimal MLP\n without PCA', 'Optimal MLP\n with PCA', 'Default MLP\n without PCA', 'VotingClassifier', 'Starting Kit\n Classifier'], '% CV score': [76, 75,65, 94, 91, 82,88, 71]})
        fig = sns.barplot(x = 'Type of classification', y = '% CV score', data = fake)
    
    
    if(donnee==2):
        plt.title('Graphique du tableau préliminaire')
        plt.ylim(0,1.1)
    
        barWidth = 0.25
        bars1 = [0.5877,0.6707,0.9996,1.0000,0.8928,1.00]
        bars2 = [0.63,0.62,0.65,0.57,0.87,0.92]
        bars3 = [0.5757,0.6673,0.6054,0.5665,0.7727,0.9366]


# Set position of bar on X axis
        r1 = np.arange(len(bars1))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
 
# Make the plot
        plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Train')
        plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='CV(+/- 0,01)')
        plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Valid(ation)')
 
# Add xticks on the middle of the group bars
        plt.xlabel('Méthode de classification', fontweight='bold')
        plt.ylabel('Score', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(bars1))], ['NaiveBayes', 'SGDC', 'RandomForest', 'DecisionTree', 'QDA','MPL'])
 
# Create legend & Show graphic
        plt.legend()




    plt.show()
    plt.close()

if __name__=="__main__":
    sns.set(font_scale=1.4,style="whitegrid") #set styling preferences
    url = "c1_input_data/perso_train.data" 
    data = data_io.read_as_df('c1_input_data/perso') #tout
    df = pd.read_csv(url ,delimiter=' ') # toutes les données sans le head, ni les labels
    labels = data.iloc[1:,-1] # label : true ou false?



    pca=PCA(n_components=200)
    x_reduced=pca.fit_transform(data)
   # print(data)
    x_reduced = pd.DataFrame(x_reduced)
    #print (x_reduced.shape)
    #data = x_reduced
    
    
    funcPCA (data)
    funcDiagbaton(1)
    funcDiagbaton(2)

    