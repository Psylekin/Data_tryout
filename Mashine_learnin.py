#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 17:35:42 2018

@author: psylekin
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from datetime import datetime

from sklearn.pipeline import Pipeline
from scipy.stats import randint as sp_randint
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

#%% Loading Data
X = pd.read_csv("/home/psylekin/AnacondaProjects/Capstone/train_values.csv", index_col = 0)
y = pd.read_csv("/home/psylekin/AnacondaProjects/Capstone/train_labels.csv", index_col = 0).as_matrix().ravel()

#%% Transforming
catlist = ["land_surface_condition", "foundation_type", "roof_type", "ground_floor_type", "other_floor_type", "position", "plan_configuration", "legal_ownership_status"]
makeCats(X,catlist)

#%% Pipeline 

scaler = KernelCenterer()
selection = Isomap()
model = RandomForestClassifier()

pipeline = Pipeline([
        ('Scaler', scaler),
        ('Selection', selection),
        ('Model', model)])

iterations = 5
parameters = dict(Selection__n_neighbors = [3,5,8,10],
              Selection__n_components = [3,5,8,10],
              Model__n_estimators = [0,1,3])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

estimator = GridSearchCV(pipeline, param_grid=parameters)
#estimator = RandomizedSearchCV(pipeline, param_distributions = parameters, n_iter = iterations)
#estimator = pipeline

estimator.fit(X,y)

y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred)
result = f1_score(y_test, y_pred, average='micro')

#%% Generate output
X_testing = pd.read_csv("/home/psylekin/AnacondaProjects/Capstone/test_values.csv", index_col = 0)
makeCats(X_testing,catlist)

y_predicted = pd.DataFrame(pipeline.predict(X_testing),columns = ["damage_grade"])
y_predicted.index = X_testing.index
y_predicted.index.names = ['building_id']
output = y_predicted
output.to_csv("/home/psylekin/AnacondaProjects/Capstone/submission.csv")


end_time = datetime.now()
print("Result: ", result)
print('Duration: {}'.format(end_time - start_time),
      "\n+++++++++++++++++++++++++++++")
