# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 22:48:52 2017

@author: Asus
"""

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
import numpy as np 
from sklearn.metrics import r2_score
import pandas as pd
from matplotlib import pyplot

data = pd.read_csv('price.csv')

res = data.cost.values
data = data.drop('cost', axis=1)

price = np.array(res)
avg = np.mean(price)

X_train, X_test, y_train, y_test = train_test_split(data, res, random_state=0,test_size=0.2)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.model_selection import cross_validate
from sklearn.cross_validation import *
from sklearn import metrics
scoring = ['explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2']
def train_and_evaluate(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    cv = KFold(X_train.shape[0], 5, shuffle=True, random_state=33)
    scores = cross_validate(clf, X_train, y_train, scoring=scoring, cv=cv, return_train_score=False)
    print ("Average using 5-fold crossvalidation: ")
    print ("test_explained_variance:{0:.4f}".format(np.mean(scores['test_explained_variance'])))
    print ("test_neg_mean_absolute_error:{0:.4f}".format(np.mean(scores['test_neg_mean_absolute_error'])))
    print ("test_neg_mean_squared_error:{0:.4f}".format(np.mean(scores['test_neg_mean_squared_error'])))
    print ("test_neg_mean_squared_log_error:{0:.4f}".format(np.mean(scores['test_neg_mean_squared_log_error'])))
    print ("test_neg_median_absolute_error:{0:.4f}".format(np.mean(scores['test_neg_median_absolute_error'])))
    print ("test_r2:{0:.4f}".format(np.mean(scores['test_r2'])))
    
def measure_performance(X_test, y_test, clf):
    y_pred = clf.predict(X_test)
    #pyplot.scatter(y_test, y_pred)
    print ("Explained_variance:{0:.4f}".format(metrics.explained_variance_score(y_test, y_pred)))
    print ("Mean_absolute_error:{0:.4f}".format(metrics.mean_absolute_error(y_test, y_pred)))
    print ("Mean_squared_error:{0:.4f}".format(metrics.mean_squared_error(y_test, y_pred)))
    print ("Mean_squared_log_error:{0:.4f}".format(metrics.mean_squared_log_error(y_test, y_pred)))
    print ("Median_absolute_error:{0:.4f}".format(metrics.median_absolute_error(y_test, y_pred)))
    print ("Coefficient of determination:{0:.4f}".format(metrics.r2_score(y_test, y_pred)))
    

#1 linear regression
clf = linear_model.LinearRegression()
train_and_evaluate(clf, X_train, y_train)
measure_performance(X_test, y_test, clf)

#3 support vector machines linear
from sklearn import svm
clf_svr = svm.SVR(kernel='linear')
train_and_evaluate(clf_svr, X_train, y_train)
measure_performance(X_test, y_test, clf_svr)

#3 support vector machines poly
clf_svr_poly = svm.SVR(kernel='poly')
train_and_evaluate(clf_svr_poly, X_train, y_train)
measure_performance(X_test, y_test, clf_svr_poly)

#4 support vector machines rbf
clf_svr_rbf = svm.SVR(kernel='rbf')
train_and_evaluate(clf_svr_rbf, X_train, y_train)
measure_performance(X_test, y_test, clf_svr_rbf)

#5 decisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
clf_dec_tree = DecisionTreeRegressor(random_state=33)
train_and_evaluate(clf_dec_tree, X_train, y_train)
measure_performance(X_test, y_test, clf_dec_tree)

#6 extra tree regressor
from sklearn import ensemble
clf_et=ensemble.ExtraTreesRegressor(n_estimators=10, random_state=42)
train_and_evaluate(clf_et, X_train, y_train)
measure_performance(X_test, y_test, clf_et)

#7 random forest
from sklearn.ensemble import RandomForestRegressor
clf_ran_for = RandomForestRegressor(n_estimators=10, max_depth=None, random_state=33)
train_and_evaluate(clf_ran_for, X_train, y_train)
measure_performance(X_test, y_test, clf_ran_for)

#8 adaboost
from sklearn.ensemble import AdaBoostRegressor
clf_ada_boost = AdaBoostRegressor()
train_and_evaluate(clf_ada_boost, X_train, y_train)
measure_performance(X_test, y_test, clf_ada_boost)
