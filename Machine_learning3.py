# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# General
from __future__ import print_function, division
import numpy as np
import pandas as pd
from datetime import datetime

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns

# Pipeline-Testing
from sklearn.pipeline import Pipeline
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV, train_test_split

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

# Scaling
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler, KernelCenterer, StandardScaler

# Mashine Learning
from sklearn.linear_model import LogisticRegression


#%% Functions
def makeCats(data, columns):
    for column in columns:
        data.loc[:,column] = data.loc[:,column].astype("category", data.loc[:,column].unique()).cat.codes

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

#%% Loading Data
start_time = datetime.now()
        
X = pd.read_csv("/home/psylekin/AnacondaProjects/data_cap/train_values.csv", index_col = 0)
y = pd.read_csv("/home/psylekin/AnacondaProjects/data_cap/train_labels.csv", index_col = 0)
X_testing = pd.read_csv("/home/psylekin/AnacondaProjects/data_cap/test_values.csv", index_col = 0)

df = X.join(y)

location = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
catlist = ["damage_grade","land_surface_condition", "foundation_type", "roof_type", "ground_floor_type", "other_floor_type", "position", "plan_configuration", "legal_ownership_status"]

makeCats(df,catlist)
makeCats(X_testing, catlist[1:])

#%% Cleaning (Ausreißer)
#Delete hight over 10
df = df.loc[df.height < 10,]

#Delete area over 200
df = df.loc[df.area < 200,]

#Delete age over 200
df = df.loc[df.age < 200,]

#Split Categories X
for x in catlist:
    new_df = pd.get_dummies(df[x],prefix = x)
    df = df.join(new_df)

#Split Categories X_testing
for x in catlist[1:]:
    new_xdf = pd.get_dummies(X_testing[x],prefix = x)
    X_testing = X_testing.join(new_xdf) #TODO: Fix this
    
end_time = datetime.now()
print("Cleaning done \n ",'Duration: {}'.format(end_time - start_time),
      "\n+++++++++++++++++++++++++++++")


#%% Splitting for damage grade 0
y_gen = df.damage_grade
y0 = df.damage_grade_0
y1 = df.damage_grade_1
y2 = df.damage_grade_2
X = df.drop(["damage_grade","damage_grade_0","damage_grade_1","damage_grade_2"], axis = 1)  

end_time = datetime.now()
print("Splitting done \n ",'Duration: {}'.format(end_time - start_time),
      "\n+++++++++++++++++++++++++++++")

#%% Pipeline for 0 or other
logistic = LogisticRegression()
pca = PCA()
pipe = Pipeline(steps=[
        ('pca', pca), 
        ('logistic', logistic)
        ])

end_time = datetime.now()    
print("Pipeline ready \n ",'Duration: {}'.format(end_time - start_time),
      "\n+++++++++++++++++++++++++++++")
#%% GridSearchCV 0
n_components = range(3,10)
Cs = np.logspace(-1, 1, 10)

estimator0 = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs),
                              scoring = "f1_micro",
                              return_train_score=True)
estimator0.fit(X, y0)

end_time = datetime.now()    
best_results0 = pd.DataFrame(estimator0.cv_results_)
print("Best Estimator 0 found! \n ",'Duration: {}'.format(end_time - start_time),
      "\n+++++++++++++++++++++++++++++")

#%% Finding best Predictor 1 VS 2

#%% Doing predictions: 0
y_pred0 = estimator0.predict(X_testing)
print("damage_grade 0 predicted! \n ",'Duration: {}'.format(end_time - start_time),
      "\n+++++++++++++++++++++++++++++")

#%% Cutting out y_pred0 == 1
X_testing = X_testing.loc[y_pred0 != 1,]
print(X_testing.shape)

#%% Predict 1 VS 2

#%% Combine y_pred0 (strong 1s) on y_pred1