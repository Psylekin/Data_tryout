# -*- coding: utf-8 -*-
"""
Learnings 

# Predicting 0      use Normalizer() & LinearSVC()
# Predicting 1      use MinMaxScaler() & RandomForestClassifier()
# Predicting 2      use MaxAbsScaler() & GradientBoostingClassifier()
# Predicting Total  use MaxAbsScaler() & GradientBoostingClassifier()
# Stack             use LinearSVC()

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
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV, train_test_split

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

# Scaling
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler, KernelCenterer, StandardScaler

# Mashine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Scoring
from sklearn.metrics import f1_score, precision_score

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
df = df.loc[df.height < 20,]

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
    X_testing = X_testing.join(new_xdf)
    
end_time = datetime.now()

#%% Splitting
# Normalizer

scaler = [Normalizer(), MaxAbsScaler(), MinMaxScaler(), StandardScaler()]
clfs = [LogisticRegression(), RandomForestClassifier(), ExtraTreesClassifier(), AdaBoostClassifier(), GradientBoostingClassifier(), LinearSVC(), KNeighborsClassifier(), DecisionTreeClassifier()]

sampleset = df.sample(1000)
df_new = df.drop(sampleset.index)

y_gen_sampleset = sampleset.damage_grade
y0_sampleset = sampleset.damage_grade_0
y1_sampleset = sampleset.damage_grade_1
y2_sampleset = sampleset.damage_grade_2
X_sampleset = sampleset.drop(["damage_grade","damage_grade_0","damage_grade_1","damage_grade_2"], axis = 1)  

y_gen = df_new.damage_grade
y0 = df_new.damage_grade_0
y1 = df_new.damage_grade_1
y2 = df_new.damage_grade_2
X = df_new.drop(["damage_grade","damage_grade_0","damage_grade_1","damage_grade_2"], axis = 1)  

#%% Settings for Search
n_jobs = 4
n_iter_search = 100
scoring = 'f1_micro'

#%% Predictor 1
start_time = datetime.now()

pipeline0 = Pipeline([("scaler", Normalizer()),("clf",LinearSVC())])
param_dist = {"clf__C": np.arange(1.5, 3.0, 0.01)}
    
estimator0 = RandomizedSearchCV(pipeline0, param_distributions=param_dist, n_iter=n_iter_search, 
                                   scoring=scoring, return_train_score=True,n_jobs = n_jobs)
estimator0.fit(X, y0)
end_time = datetime.now()    
y_predicted = estimator0.predict(X)
y_predicted_sampleset = estimator0.predict(X_sampleset)
       
print("Estimator 1\nOutside: \t", f1_score(y0_sampleset, y_predicted_sampleset, average= 'micro'))
print("Inside: \t", f1_score(y0, y_predicted, average= 'micro'))

end_time = datetime.now()
print("Duration: \t{}".format(end_time - start_time), "\n+++++++++++++++++++++++++++++")

#%% Predictor 2
start_time = datetime.now()

pipeline1 = Pipeline([("scaler", MinMaxScaler()),("clf",RandomForestClassifier())])

param_dist = {"clf__n_estimators": sp_randint(1,20),
              "clf__criterion": ["gini", "entropy"],
              "clf__max_depth": sp_randint(5,15),
              "clf__max_features": sp_randint(5,25)
              }
    
estimator1 = RandomizedSearchCV(pipeline1, param_distributions=param_dist,
                                   n_iter=n_iter_search, scoring=scoring, return_train_score=True, n_jobs = n_jobs)
estimator1.fit(X, y1)

y_predicted = estimator1.predict(X)
y_predicted_sampleset = estimator1.predict(X_sampleset)

print("Estimator 2\nOutside: \t", f1_score(y1_sampleset, y_predicted_sampleset, average= 'micro'))
print("Inside: \t", f1_score(y1, y_predicted, average= 'micro'))

end_time = datetime.now()
print("Duration: \t{}".format(end_time - start_time), "\n+++++++++++++++++++++++++++++")

#%% Predictor 3
start_time = datetime.now()

param_dist = {"clf__learning_rate": np.arange(0.11,3,0.01),
              "clf__loss": ["deviance", "exponential"],
              "clf__n_estimators": sp_randint(10,150),
              "clf__max_depth": sp_randint(1,7)
              }

pipeline2 = Pipeline([("scaler", MaxAbsScaler()),("clf",GradientBoostingClassifier() )])

estimator2 = RandomizedSearchCV(pipeline2, param_distributions=param_dist,
                                   n_iter=n_iter_search, scoring=scoring, return_train_score=True, n_jobs = n_jobs)


estimator2.fit(X, y2)
y_predicted = estimator2.predict(X)
y_predicted_sampleset = estimator2.predict(X_sampleset)

print("Estimator 3\nOutside: \t", f1_score(y2_sampleset, y_predicted_sampleset, average= 'micro'))
print("Inside: \t", f1_score(y2, y_predicted, average= 'micro'))


end_time = datetime.now()
print("Duration: \t{}".format(end_time - start_time), "\n+++++++++++++++++++++++++++++")

#%% Predictor All
start_time = datetime.now()

param_dist = {"clf__learning_rate": np.arange(0.01,5,0.01),
              "clf__n_estimators": sp_randint(80,200),
              "clf__max_depth": sp_randint(1,8)
              }


pipeline_all = Pipeline([("scaler", MaxAbsScaler()), ("clf",GradientBoostingClassifier())])

estimator_all = RandomizedSearchCV(pipeline_all, param_distributions=param_dist,
                                   n_iter=n_iter_search, scoring=scoring, return_train_score=True, n_jobs = n_jobs)

estimator_all.fit(X, y_gen)
y_predicted = estimator_all.predict(X)
y_predicted_sampleset = estimator_all.predict(X_sampleset)

print("Estimator All\nOutside: \t", f1_score(y_gen_sampleset, y_predicted_sampleset, average= 'micro'))
print("Inside: \t", f1_score(y_gen, y_predicted, average= 'micro'))

end_time = datetime.now()
print("Duration: \t{}".format(end_time - start_time), "\n+++++++++++++++++++++++++++++")

#%% Predict 0 VS rest
y_bench0 = estimator0.predict(X)
y_predicted_sampleset0 = estimator0.predict(X_sampleset)
y_pred0 = estimator0.predict(X_testing)

#%% Predict 1 VS rest
y_bench1 = estimator1.predict(X)
y_predicted_sampleset1 = estimator1.predict(X_sampleset)
y_pred1 = estimator1.predict(X_testing)

#%% Predict 2 VS rest

y_bench2 = estimator2.predict(X)
y_predicted_sampleset2 = estimator2.predict(X_sampleset)
y_pred2 = estimator2.predict(X_testing)

#%% Predict All
y_bench_all = estimator_all.predict(X)
y_predicted_sampleset_all = estimator_all.predict(X_sampleset)
y_pred_all = estimator_all.predict(X_testing)

#%% Stack

y_bench_stack = pd.DataFrame([y_bench0,y_bench1,y_bench2,y_bench_all+1]).T
y_sampleset_stack = pd.DataFrame([y_predicted_sampleset0,y_predicted_sampleset1,y_predicted_sampleset2,y_predicted_sampleset_all+1]).T
y_pred_stack = pd.DataFrame([y_pred0,y_pred1,y_pred2,y_pred_all+1]).T

# Normalizer
    
pipeline_stack = Pipeline([("clf", LinearSVC())])

param_dist = {"clf__C": np.arange(0.1, 2.0, 0.01)}
    
estimator_stack = RandomizedSearchCV(pipeline_stack, param_distributions=param_dist,
                                   n_iter=n_iter_search, 
                                   scoring=scoring, 
                                   return_train_score=True,
                                   n_jobs = 5)
estimator_stack.fit(y_bench_stack, y_gen)

y_predicted = estimator_stack.predict(y_bench_stack)
y_predicted_sampleset = estimator_stack.predict(y_sampleset_stack)

print("Estimator Stack\nOutside\nF1 Score : \t", f1_score(y_gen_sampleset, y_predicted_sampleset, average= 'micro'))

#%% Stupid Stack
"""
for x in y_predicted_sampleset0.index:
    if y_predicted_sampleset0[x] != 

y_predicted_sampleset 


print("Estimator Stupid\nOutside\nF1 Score : \t", f1_score(y_gen_sampleset, y_predicted_sampleset, average= 'micro'))
"""


#%% Final prediction

y_final = estimator_stack.predict(y_pred_stack)


#%% Output

output = pd.DataFrame(y_final+1, columns = ["damage_grade"])
output.index = X_testing.index
output.to_csv("/home/psylekin/AnacondaProjects/data_cap/submission.csv")

print("Output completed!")


#%% Learnings

print("\n1\t",estimator0.best_params_)
print("2\t", estimator1.best_params_)
print("3\t", estimator2.best_params_)
print("All\t", estimator_all.best_params_)
print("Stack\t", estimator_stack.best_params_)
