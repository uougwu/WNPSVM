# determine the parameters from training data - baseline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,balanced_accuracy_score

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import random

def parameters(meta,Kingry):

    Lungs = Kingry['A']
    zb = Kingry['yb'].flatten()
    yb = zb/np.linalg.norm(zb)
    Aproj = np.abs(np.dot(Lungs,yb))
    ind1 = Aproj.argsort()[::-1]
    ALB = Lungs[ind1,:]    
    N = 235
    ALLB = ALB[0:N,:]


    #ALLB = StandardScaler().fit_transform(ALLB[:,0:48].T) # Schu4
    ALLB = StandardScaler().fit_transform(np.concatenate((ALLB[:,0:24].T,ALLB[:,48:72].T))) # LVS
    ytrain = np.concatenate((np.array([0]*24),np.array([1]*24)))

    # # define models and parameters
    model = GaussianNB()
    param_grid = {'var_smoothing': [1,10,1e-2,1e-3,1e-1]}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=10)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(ALLB, ytrain)
    # summarize results
    print("NB Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    #define models and parameters
    model = KNeighborsClassifier()
    param_grid = {'n_neighbors': [5,3,7,10]
                }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=10)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(ALLB, ytrain)
    # summarize results
    print("KNN Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # # define models and parameters
    model = LogisticRegression(random_state=10)
    param_grid = {'C' : [1, 10, 100, 0.1, 0.01],
                'solver':['lbfgs','liblinear']}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=10)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(ALLB, ytrain)
    # summarize results
    print("LR Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # define models and parameters
    model = svm.SVC()
    param_grid = {'C': [1,50,10,100,1e-2,1e-3,1e-1],
                'kernel':['linear']}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=10)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(ALLB, ytrain)
    # summarize results
    print("SVM Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

        
    # define models and parameters
    model = RandomForestClassifier(random_state=10)
    param_grid = {
        'max_features': ['sqrt', 'log2'],
        'n_estimators': [100, 200, 50, 300],'criterion':['gini', 'entropy']}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=10)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(ALLB, ytrain)
    # summarize results
    print("RF Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # define models and parameters
    model = tree.DecisionTreeClassifier(random_state=10)
    param_grid = {
        'max_depth': [2, 3, 5, 10],
        'min_samples_leaf': [1,5, 10, 15],
        'criterion':['gini', 'entropy']}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=10)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(ALLB, ytrain)
    # summarize results
    print("DT Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        
    # # define models and parameters
    model = GradientBoostingClassifier(random_state=10)
    n_estimators = [100,200,300]
    learning_rate = [0.1,0.01,0.001,1]
    max_depth = [3,4,2,5]
    # define grid search
    grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=10)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(ALLB, ytrain)
    # summarize results
    print("GB Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        

    # # define models and parameters
    model = AdaBoostClassifier(random_state=10)
    n_estimators = [50,100,200,300]
    learning_rate = [1,0.1,0.01,0.001]
    # define grid search
    grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=10)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(ALLB, ytrain)
    # summarize results
    print("AB Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        
    # define models and parameters
    model = XGBClassifier(random_state=10)
    eta = [0.3,0.2,0.1,0.01,0.5,0.001] 
    max_depth = [6,4,5,7] 
    # define grid search
    grid = dict(learning_rate=eta, max_depth=max_depth)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=10)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(ALLB, ytrain)
    # summarize results
    print("XGB Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))