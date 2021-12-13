# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 09:57:24 2021

@author: Administrator
"""
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestRegressor
import numpy as np

data = pd.read_csv('D:/s-sn/ASCII1/evidence1.csv',encoding='gbk')



x = data.drop('status',axis=1)
y = data.status
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state =2018, shuffle = True)

'''

rf = RandomForestClassifier()
#不做优化的结果
print(np.mean(cross_val_score(rf,X_train,y_train,scoring="accuracy",cv=20)))
'''

def rf_cv(n_estimators, min_samples_split, max_depth, max_features):
    val = cross_val_score(RandomForestClassifier(n_estimators=int(n_estimators),
                          min_samples_split=int(min_samples_split),
                          max_depth = int(max_depth),
                          max_features = min(max_features,0.999),
                          random_state = 2),
            X_train,y_train,scoring="accuracy",cv=5).mean()
    return val
#贝叶斯优化
rf_bo = BayesianOptimization(rf_cv,
                             {
                                 "n_estimators":(10,250),
                                 "min_samples_split":(2,25),
                                 "max_features":(0.1,0.999),
                                 "max_depth":(5,15)
                             })
#开始优化
num_iter = 25
init_points = 5
rf_bo.maximize(init_points=init_points,n_iter=num_iter)
#显示优化结果
rf_bo.res["max"]
#附近搜索（已经有不错的参数值的时候）


rf_bo.explore(
    {'n_estimators': [10, 100, 200],
     'min_samples_split': [2, 10, 20],
     'max_features': [0.1, 0.5, 0.9],
     'max_depth': [5, 10, 15]
    })

#验证优化后参数的结果
rf = RandomForestClassifier(max_depth=5, max_features=0.432, min_samples_split=2, n_estimators=190)
np.mean(cross_val_score(rf, X_train, y_train, cv=20, scoring='roc_auc'))
