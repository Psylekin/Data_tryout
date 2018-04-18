# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import print_function, division



import numpy as np
import seaborn as sns
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
from sklearn.metrics import f1_score, precision_score
from sklearn.linear_model import LogisticRegression


import pandas as pd
import matplotlib.pyplot as plt

def makeCats(X, columns):
    for column in columns:
        X.loc[:,column] = X.loc[:,column].astype("category", X.loc[:,column].unique()).cat.codes

def correlations(target):
    correlations = df.corr()
    pos_correlations = correlations.loc[correlations.loc[:,target] > 0.19 ,target]
    neg_correlations = correlations.loc[correlations.loc[:,target] < -0.19 ,target]
    correlates = pos_correlations.append(neg_correlations)
    return correlates

def series_to_latex(data):
    data = data.sort_values()
    for x in data.index:
        print(x.replace("_"," ")," & ",round(data.loc[x],2), "\\\ ")


X = pd.read_csv("/home/psylekin/AnacondaProjects/data_cap/train_values.csv", index_col = 0)
y = pd.read_csv("/home/psylekin/AnacondaProjects/data_cap/train_labels.csv", index_col = 0)

df = X.join(y)

location = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
catlist = ["damage_grade","land_surface_condition", "foundation_type", "roof_type", "ground_floor_type", "other_floor_type", "position", "plan_configuration", "legal_ownership_status"]
makeCats(df,catlist)

print("Import done")
#%% Cleaning (Ausreißer)
#Delete hight over 10
df = df.loc[df.height < 10,]

#Delte area over 200
df = df.loc[df.area < 200,]

#Delte age over 200
df = df.loc[df.age < 200,]


#Split Categories
for x in catlist:
    new_df = pd.get_dummies(df[x],prefix = x)
    df = df.join(new_df)
print("Cleaning done")


#%% Splitting for damage grade 0
y0 = df.damage_grade_0
y1 = df.damage_grade_1
y2 = df.damage_grade_2
X = df.drop(["damage_grade","damage_grade_0","damage_grade_1","damage_grade_2"], axis = 1)  
print("Splitting done")

#%%Pipeline
logistic = LogisticRegression()

pca = PCA()
pipe = Pipeline(steps=[
        ('pca', pca), 
        ('logistic', logistic)
        ])
    
print("Pipeline ready")
#%% GridSearchCV 0
n_components = range(3,10)
Cs = np.logspace(-1, 1, 10)

estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs),
                              scoring = "f1_micro",
                              return_train_score=True)
estimator.fit(X, y0)
best_results0 = pd.DataFrame(estimator.cv_results_)
print("Results 0 are ready!")

prediction = estimator.predict(X)

print("Estimator Total\nF1 Score: \t", f1_score(y0, prediction, average='micro'))
print("Precision : \t", precision_score(y0, prediction, average = 'micro'))


#%% Importance picker
"""
clf = RandomForestClassifier()
clf.fit(X,y)

importance = clf.feature_importances_ > 0.05
df_small = df.loc[:,importance]
"""

#%% Correlatives
"""
correlations_base = correlations("damage_grade")
correlations0 = correlations("damage_grade_0")
correlations1 = correlations("damage_grade_1")
correlations2 = correlations("damage_grade_2")

print("Correlations done")
"""