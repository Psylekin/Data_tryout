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
y_label = pd.read_csv("/home/psylekin/AnacondaProjects/data_cap/train_labels.csv", index_col = 0)
print(X.shape)

new_df = pd.DataFrame()
num_data_list = ["count_families","count_floors_pre_eq", "height", "area", "age"]
for x in num_data_list: 
    new_df = pd.concat([new_df,X[x].describe()[1:]],axis = 1)
    new_df = new_df
    
#for x in X.columns: 
#    print("\n",X[x].value_counts())

#%%
    
targets = ["legal_ownership_status","plan_configuration",
           #"geo_level_1_id","geo_level_2_id","geo_level_3_id",
           "land_surface_condition",
           "foundation_type","roof_type","ground_floor_type","other_floor_type",
           "position","plan_configuration"]
for target in targets:
    print(X[target].name, "&& \\\ ")
    for y in X[target].value_counts().index: 
            print(y, "&", X[target].value_counts()[y], "\\\ ")
    print("&& \\\ ")
    
#%%
    
X = X.loc[X.height < 20,]

#Delete area over 200
X = X.loc[X.area < 200,]

#Delete age over 200
X = X.loc[X.age < 200,]

#%%
x = 0
z = 0
y = 0
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 6), sharey=True)

for x in range(3,5):
    if x == 3: 
        z = 1
        y = 0
    axes[y].boxplot(X[num_data_list[x]])
    axes[y].set_title(num_data_list[x])
    y += 1

plt.show()

#for x in range(len(num_data_list)):     
#    plt.title(num_data_list[x])
    

#%%
catlist = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',"land_surface_condition", "foundation_type", "roof_type", "ground_floor_type", "other_floor_type", "position", "plan_configuration", "legal_ownership_status"]
makeCats(X,catlist)


df = pd.concat([X,y_label], axis = 1)

new_df = pd.get_dummies(df["damage_grade"],prefix = "damage_grade")
df = df.join(new_df)

x = "damage_grade"

correlations = df.corr()
list1 = correlations.loc[correlations[x] >= 0.2,x].index.tolist()
list2 = correlations.loc[correlations[x] <= -0.2,x].index.tolist()
list1 = list1+list2

df_corrs = df.loc[:,list1]
df_corrs = df_corrs.drop([
        "damage_grade_1",
        "damage_grade_2",
        "damage_grade_3", 
        #"damage_grade"
        ], axis = 1)
correlations = df_corrs.corr()

sns.heatmap(correlations, 
        xticklabels=correlations.columns,
        yticklabels=correlations.columns)

print(correlations[x])
