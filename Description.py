# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv("/home/psylekin/AnacondaProjects/Capstone/train_values.csv", index_col = 0)
print(X.height.describe(), "\n", X.height.median())

y = pd.read_csv("/home/psylekin/AnacondaProjects/Capstone/train_labels.csv", index_col = 0)
y_dist = y.damage_grade.value_counts()
plt.bar(y_dist.index, y_dist.values)
plt.show()

df = X.join(y,how = "outer")

dm_1_mean = df.loc[df.damage_grade == 1,"age"].mean()
dm_2_mean = df.loc[df.damage_grade == 2,"age"].mean()
dm_3_mean = df.loc[df.damage_grade == 3,"age"].mean()

print("Age Mean\n1:", dm_1_mean, "\n2:", dm_2_mean, "\n3:", dm_3_mean)

dm_1_mean = df.loc[df.damage_grade == 1,"height"].median()
dm_2_mean = df.loc[df.damage_grade == 2,"height"].median()
dm_3_mean = df.loc[df.damage_grade == 3,"height"].median()

print("Height Mean\nDurchschnitt: ", df.height.mean(), "\n1:", dm_1_mean, "\n2:", dm_2_mean, "\n3:", dm_3_mean)


print("Area\n", df.loc[df.area > df.area.mean(),"damage_grade"].value_counts())