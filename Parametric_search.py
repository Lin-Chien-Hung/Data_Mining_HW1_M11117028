import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

def Parametric_search():

    adult = pd.read_csv('adult.data')
    adult_test = pd.read_csv('adult.test')

    X = adult.iloc[:,:-1]
    y = adult['income']

    test_X = adult_test.iloc[:,:-1]
    test_y = adult_test['income']

    #=======================================================================================================================================================

    #labelencoder_X = LabelEncoder()
    X = X.apply(LabelEncoder().fit_transform)
    test_X = test_X.apply(LabelEncoder().fit_transform)

    #=======================================================================================================================================================
    #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    # 選出效果最好的參數
    parameters={
                'criterion':['gini','entropy'],
                'max_depth':range(1,20),
                'min_samples_split':range(2,20),
                'min_samples_leaf':range(1,20)
                }

    clf = tree.DecisionTreeClassifier()
    grid_search = GridSearchCV(clf, parameters, scoring='accuracy',cv=5,n_jobs=-1)
    grid_search.fit(X,y)

    best_estimator = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_param = grid_search.best_params_

    print(best_estimator)
    print(best_score)
    print(best_param)

if __name__ == '__main__':
    Parametric_search()
