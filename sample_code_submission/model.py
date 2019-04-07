'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn import tree

#from sklearn.model_selection import GridSearchCV
#from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
#from sklearn.neighbors import KNeighborsClassifier 
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from PREPROCESSSSSSING import Preprocessor
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
#from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

class model (BaseEstimator):
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        #self.model = clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')  
        #self.model = clf = GaussianNB()
        #self.model = clf = KNeighborsClassifier()
        #self.model = clf = QuadraticDiscriminantAnalysis()
        #self.model = clf = RandomForestClassifier(n_estimators= 80 , max_depth= 20, max_features= 'sqrt')

       # self.model = clf = Pipeline([('preprocessing', Preprocessor()),('classification', MLPClassifier(hidden_layer_sizes=(100, 50, 100, 50, 20),max_iter=1500,solver='adam', learning_rate='adaptive', activation='relu'))])
       
              
        fancy_classifier1 = Pipeline([('preprocessing', Preprocessor()),('classification', RandomForestClassifier(n_estimators= 80 , max_depth= 20, max_features= 'sqrt'))])
        fancy_classifier2 = Pipeline([('preprocessing', Preprocessor()),('classification', MLPClassifier(hidden_layer_sizes=(200,100,50,20),max_iter=1500,solver='adam', learning_rate='invscaling', activation='relu'))])
#       
#					
        self.model = clf = VotingClassifier(estimators=[
					('Fancy Classifier1', fancy_classifier1),
                    ('Fancy Classifier2', fancy_classifier2),
                    
                    
                    ],
					voting='soft')  
    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''
        
        self.num_train_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1]
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = y.shape[0]
        if y.ndim>1: self.num_labels = y.shape[1]
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        self.is_trained=True
        self.model = self.model.fit(X, y)
        

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        num_test_samples = X.shape[0]
        if X.ndim>1: num_feat = X.shape[1]
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        y = self.model.predict_proba(X)
        
        # If you uncomment the next line, you get pretty good results for the Iris data :-)
        #y = np.round(X[:,3])
       
        return y[:,1]

    def save(self, path="./"):
        pickle.dump(self.model, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
