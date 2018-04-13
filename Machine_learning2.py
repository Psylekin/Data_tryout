#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 17:35:42 2018

@author: psylekin
"""

# Authors: Robert McGibbon, Joel Nothman, Guillaume Lemaitre

from __future__ import print_function, division



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from datetime import datetime

from sklearn.pipeline import Pipeline
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
# Scaling
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler, KernelCenterer, StandardScaler
# Mashine Learning
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
# Scoring
from sklearn.metrics import f1_score


#%% Functions

def transform(X_train, X_test, model):
    X_train = model.fit_transform(X_train)
    X_test = model.transform(X_test)
    return X_train, X_test

def makeCats(X, columns):
    for column in columns:
        X.loc[:,column] = X.loc[:,column].astype("category", X.loc[:,column].unique()).cat.codes

#%% Zeitmessung
start_time = datetime.now()

#%%Pipeline
reductor = PCA(n_components = 8)
scaler = KernelCenterer()
classifier = RandomForestClassifier()


pipe = Pipeline([
    ('reduce_dim', reductor),
    ('Scaler', scaler),
    ('classify', classifier)
])
    
#%% RandomizedSearchCV
    
n_components = [3,5,7,9,11,13]
n_estimators = [5,7,9,11,13,15]
n_neighbors = [3,5,7,9]
C_values = [0.001, 0.01,0.1,1]

# specify parameters and distributions to sample from
param_dist = {
        'reduce_dim': [PCA()],
        'classify' : [RandomForestClassifier()],
        'classify__n_estimators' : randint(1,20),
        'classify__max_depth' : randint(1,10)
        }

# run randomized search
n_iter_search = 50
n_jobs = 10
grid = RandomizedSearchCV(pipe, n_jobs = n_jobs, scoring = "f1_micro", param_distributions = param_dist, n_iter = n_iter_search)

#%%GridSearchCV
n_components = [3,5,7,9,11,13]
n_estimators = [5,7,9,11,13,15]
n_neighbors = [3,5,7,9]
C_values = [0.5,0.8,1,1.3,1.5]

param_grid = [{
        'reduce_dim': [PCA()],
        'reduce_dim__n_components': n_components,
        'classify': [SVC()],
        'classify__kernel' : ["linear"],
        'classify__C' : C_values
        }]
"""
{
        'reduce_dim': [PCA()],
        'reduce_dim__n_components': n_components,
        'classify': [KNeighborsClassifier()],
        'classify__n_neighbors' : n_neighbors
        }

{
        'reduce_dim': [PCA()],
        'reduce_dim__n_components': n_components,
        'classify': [RandomForestClassifier()],
        'classify__n_estimators' : n_estimators
        }
{
    'reduce_dim': [PCA()],
    'reduce_dim__n_components': n_components,
    'classify': [SVC()],
    'classify__kernel' : ["rbf"]
    }
"""

# grid = GridSearchCV(pipe, cv=3, n_jobs=1, scoring = "f1_micro", param_grid = param_grid)

#%% Loading Data & Editing Data
X = pd.read_csv("/home/psylekin/AnacondaProjects/Capstone/train_values.csv", index_col = 0)
y = pd.read_csv("/home/psylekin/AnacondaProjects/Capstone/train_labels.csv", index_col = 0).as_matrix().ravel()
catlist = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',"land_surface_condition", "foundation_type", "roof_type", "ground_floor_type", "other_floor_type", "position", "plan_configuration", "legal_ownership_status"]
makeCats(X,catlist)

#%% Pipelining
grid.fit(X, y)
results = pd.DataFrame(grid.cv_results_)

#%% Generate output

X_testing = pd.read_csv("/home/psylekin/AnacondaProjects/Capstone/test_values.csv", index_col = 0)
makeCats(X_testing,catlist)

y_predicted = pd.DataFrame(grid.predict(X_testing),columns = ["damage_grade"])
y_predicted.index = X_testing.index
y_predicted.index.names = ['building_id']
y_predicted.to_csv("/home/psylekin/AnacondaProjects/Capstone/submission.csv")


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time),
      "\n+++++++++++++++++++++++++++++")

output = "\nScore:", str(grid.best_score_), "\n", 'Duration: {}'.format(end_time - start_time),"\n+++\n"
with open("/home/psylekin/AnacondaProjects/Capstone/log.txt", 'a') as outfile:
    for x in output:
        outfile.write(x)
    for key in grid.best_params_:
        outfile.write(str(grid.best_params_[key]))
    
