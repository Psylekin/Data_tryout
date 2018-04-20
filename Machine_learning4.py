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
    X_testing = X_testing.join(new_xdf)
    
end_time = datetime.now()
#print("Cleaning done \n ",'Duration: \t{}'.format(end_time - start_time),
#      "\n+++++++++++++++++++++++++++++")

#%% Splitting
start_time = datetime.now()

best_f1_scale = []
best_f1_clf = []
best_prec_scale = []
best_prec_clf = []

# Normalizer

scaler = [Normalizer(), MaxAbsScaler(), MinMaxScaler(), StandardScaler()]
clfs = [LogisticRegression(), RandomForestClassifier(), ExtraTreesClassifier(), AdaBoostClassifier(), GradientBoostingClassifier(), LinearSVC(), KNeighborsClassifier(), DecisionTreeClassifier()]
n_iter_search = 1
scoring = 'f1_micro'


for x in range(10):
    print(x)
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
    
    searching_for = y0
    compare_to = y0_sampleset
    
    best_pipe_f1 = (0,0,0)
    best_pipe_prec = (0,0,0)
    
    for i,x in enumerate(scaler):
        for j,y in enumerate(clfs):
            start_time = datetime.now()
            pipeline0 = Pipeline([("scaler", x),
                                 ("clf",y)])
            
            param_dist = {
                          #"clf__C": sp_randint(1,10)
                          }
                
            estimator0 = RandomizedSearchCV(pipeline0, param_distributions=param_dist,
                                               n_iter=n_iter_search, 
                                               scoring=scoring, 
                                               return_train_score=True,
                                               n_jobs = 5)
            estimator0.fit(X, searching_for)
            end_time = datetime.now()    
            y_predicted = estimator0.predict(X)
            y_predicted_sampleset = estimator0.predict(X_sampleset)
            
            if f1_score(compare_to, y_predicted_sampleset, average='micro') > best_pipe_f1[0]:
                best_pipe_f1 = f1_score(compare_to, y_predicted_sampleset, average='micro'), i, j
    
            if precision_score(compare_to, y_predicted_sampleset, average='micro') > best_pipe_prec[0]:
                best_pipe_prec = precision_score(compare_to, y_predicted_sampleset, average='micro'), i, j
    
    #print("\nBest Solution F1\n")
    best_f1_scale.append(best_pipe_f1[1])
    best_f1_clf.append(best_pipe_f1[2])
    
    #print("\nBest Solution Prec\n")
    best_prec_scale.append(best_pipe_prec[1])
    best_prec_clf.append(best_pipe_prec[2])

# Predicting 0      use Normalizer() & LinearSVC()
# Predicting 1      use MinMaxScaler() & RandomForestClassifier()
# Predicting 2      use MaxAbsScaler() & GradientBoostingClassifier()
# Predicting Total  use MaxAbsScaler() & GradientBoostingClassifier()

print("sca:\t", best_prec_scale)        
print("clf:\t", best_prec_clf)
print(pd.Series(best_prec_scale).value_counts())
print(pd.Series(best_prec_clf).value_counts())

end_time = datetime.now()
print("Duration: \t{}".format(end_time - start_time), "\n+++++++++++++++++++++++++++++")

#%% Predictor 1

pipeline1 = Pipeline([("scaler", MinMaxScaler()),
                     ("clf",DecisionTreeClassifier() )])

param_dist = {"clf__criterion":["gini", "entropy"],
              "clf__max_depth": sp_randint(1,10),
              "clf__max_features": sp_randint(1,20)
              }
    
estimator1 = RandomizedSearchCV(pipeline1, param_distributions=param_dist,
                                   n_iter=n_iter_search, scoring=scoring, return_train_score=True)
estimator1.fit(X, y1)

y_predicted = estimator1.predict(X)
y_predicted_sampleset = estimator1.predict(X_sampleset)

print("Estimator 1\nF1 Score : \t", f1_score(y1_sampleset, y_predicted_sampleset))
#print("Precision: \t", precision_score(y1_sampleset, y_predicted_sampleset))

#print("\nEstimator 1\nF1 Score: \t", f1_score(y1, y_predicted))
#print("Precision: \t", precision_score(y1, y_predicted))

end_time = datetime.now() 
print("Duration: \t{}".format(end_time - start_time), "\n+++++++++++++++++++++++++++++")

#%% Predictor 2

param_dist = {"clf__criterion":["gini", "entropy"],
              "clf__max_depth": sp_randint(1,10),
              "clf__max_features": sp_randint(1,20)
              }

pipeline2 = Pipeline([("scaler", MinMaxScaler()),
                     ("clf",DecisionTreeClassifier() )])

estimator2 = RandomizedSearchCV(pipeline2, param_distributions=param_dist,
                                   n_iter=n_iter_search, scoring='f1_micro', return_train_score=True)


estimator2.fit(X, y2)
y_predicted = estimator2.predict(X)
y_predicted_sampleset = estimator2.predict(X_sampleset)

print("Estimator 2\nF1 Score : \t", f1_score(y2_sampleset, y_predicted_sampleset))
#print("Precision: \t", precision_score(y2_sampleset, y_predicted_sampleset))

end_time = datetime.now()
#print("\nEstimator 2\nF1 Score: \t", f1_score(y2, y_predicted))
#print("Precision: \t", precision_score(y2, y_predicted))

print("Duration: \t{}".format(end_time - start_time), "\n+++++++++++++++++++++++++++++")

#%% Predict 0 VS rest
y_pred0 = estimator0.predict(X_testing)
y_bench0 = estimator0.predict(X)
y_predicted_sampleset0 = estimator0.predict(X_sampleset)

#print("damage_grade 0 predicted! \n ",'Duration: \t{}'.format(end_time - start_time), "\n+++++++++++++++++++++++++++++")

#%% Predict 1 VS rest
y_pred1 = estimator1.predict(X_testing)
y_bench1 = estimator1.predict(X)
y_predicted_sampleset1 = estimator1.predict(X_sampleset)

#print("damage_grade 1 predicted! \n ",'Duration: \t{}'.format(end_time - start_time), "\n+++++++++++++++++++++++++++++")

#%% Predict 2 VS rest
y_pred2 = estimator2.predict(X_testing)
y_bench2 = estimator2.predict(X)
y_predicted_sampleset2 = estimator2.predict(X_sampleset)

#print("damage_grade 2 predicted! \n ",'Duration: \t{}'.format(end_time - start_time), "\n+++++++++++++++++++++++++++++")


#%% Combine y_pred0 (strong 1s) on y_pred1
y_final = np.ones(len(y_pred0)) + 1

for x in range(len(y_final)):
    if y_pred0[x] == 1:
        y_final[x] = 1
        
for x in range(len(y_final)):
    if y_pred1[x] == 1:
        y_final[x] = 2  
        
for x in range(len(y_final)):
    if y_pred2[x] == 1:
        y_final[x] = 3


    
y_final = y_final.astype("int")
#%% Testing
        
y_bench = np.ones(len(y_bench0)) + 1

for x in range(len(y_bench0)):
    if y_bench0[x] == 1:
        y_bench[x] = 1

for x in range(len(y_bench0)):
    if y_bench1[x] == 1:
        y_bench[x] = 2     

for x in range(len(y_bench0)):
    if y_bench2[x] == 1:
        y_bench[x] = 3

y_bench = y_bench.astype("int") - 1

print("Estimator Total\nF1 Intern: \t", f1_score(y_gen, y_bench, average='micro'))
#print("Precision : \t", precision_score(y_gen, y_bench, average = 'micro'),"\n+++++++++++++++++++++++++++++")

#%% Testing2


y_bench_sample = np.ones(len(y_predicted_sampleset0)) + 1
"""
def put_predictions_in(prediction, sample, value):
    if sample[x] == 1:
        prediction[x] = value

y_bench_sample = put_predictions_in(y_predicted_sampleset0, 1 )
"""

for x in range(len(y_predicted_sampleset0)):
    if y_predicted_sampleset0[x] == 1:
        y_bench_sample[x] = 1

for x in range(len(y_predicted_sampleset0)):
    if y_predicted_sampleset1[x] == 1:
        y_bench_sample[x] = 2

for x in range(len(y_predicted_sampleset0)):
    if y_predicted_sampleset2[x] == 1:
        y_bench_sample[x] = 3

y_bench_sample = y_bench_sample.astype("int") - 1

print("\nF1 Extern: \t", f1_score(y_gen_sampleset, y_bench_sample, average='micro'))
#print("Precision : \t", precision_score(y_gen_sampleset, y_bench_sample, average = 'micro'))

#%% Results

#print("\n0\t", estimator0.best_params_)
#print("1\t", estimator1.best_params_)
#print("2\t", estimator2.best_params_)

#%% output

output = pd.DataFrame(y_final, columns = ["damage_grade"])
output.index = X_testing.index
output.to_csv("/home/psylekin/AnacondaProjects/data_cap/submission.csv")

print("Output completed!")


"""
         0      1       2       Total
         .25    .68     .399    .608
"""