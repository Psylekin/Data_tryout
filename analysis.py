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


#%% Loading Data & Editing Data
X = pd.read_csv("/home/psylekin/AnacondaProjects/data_cap/train_values.csv", index_col = 0)
y = pd.read_csv("/home/psylekin/AnacondaProjects/data_cap/train_labels.csv", index_col = 0)
catlist = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',"land_surface_condition", "foundation_type", "roof_type", "ground_floor_type", "other_floor_type", "position", "plan_configuration", "legal_ownership_status"]
makeCats(X,catlist)

df = pd.concat([X,y], axis = 1)

correlations = df.corr()
list1 = correlations.loc[correlations.damage_grade >= 0.15,"damage_grade"].index.tolist()
list2 = correlations.loc[correlations.damage_grade <= -0.15,"damage_grade"].index.tolist()
list1 = list1+list2

df = df.loc[:,list1]

x=3
df.iloc[:,x].value_counts().plot.bar(label = df.iloc[:,x].name)