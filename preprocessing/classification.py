import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from statistics import stdev
from statistics import mean
import re
import os
import sys
import numpy
import scipy
import matplotlib
import sklearn
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer
from itertools import repeat

def make_sample(df_group1, df_group2):
    'erstelle ein sample mit zwei gleich großen Gruppen aus ursprünglich unterschiedlich großen Gruppen'
    #bestimme kleineres sample
    if df_group1.shape[0] <= df_group2.shape[0]:
        sample = pd.concat([df_group1, df_group2.sample(n= df_group1.shape[0])])
    else:
        sample = pd.concat([df_group2, df_group1.sample(n= df_group2.shape[0])])
    # change order of rows, so that the groups are merged (otherwise accuracy scoring will always yield the same results)
    sample = sample.sample(frac=1)
    return sample

def cv_LogReg_repeat(df_group1, df_group2, n):
    'erstelle ein sample mit zwei gleich großen Gruppen aus ursprünglich unterschiedlich großen Gruppen'
    # bestimme kleineres sample
    results_list = []
    for i in range(n):
        if df_group1.shape[0] <= df_group2.shape[0]:
            sample = pd.concat([df_group1, df_group2.sample(n=df_group1.shape[0])])
        else:
            sample = pd.concat([df_group2, df_group1.sample(n=df_group2.shape[0])])

        # randomisiere die genre label zuordnungen im sample'
        # sample['genre_label'] = sample['genre_label'].sample(frac=1).tolist()
        array = sample.values
        X = array[:, 0:(sample.shape[1] - 1)]
        Y = array[:, (sample.shape[1] - 1)]
        validation_size = 0.20
        seed = 7
        scoring = 'accuracy'
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                        random_state=seed)
        model = LogisticRegression(solver='liblinear', multi_class='ovr')
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        msg = (cv_results.mean(), cv_results.std())
        results_list.append(msg)
    return results_list

def mean_cv_LogReg_repeat(result_list_of_tuples):
    'returns the mean of the repeated cross validations. Input list of tuplesfrom function cv_LogReg_repeat, Output: mean values'
    means = np.asarray([tup[0] for tup in result_list_of_tuples])
    return means.mean()

def stdev_cv_LogReg_repeat(result_list_of_tuples):
    'returns the std of the repeated cross validations. Input list of tuples from function cv_LogReg_repeat, Output: std values'
    stdevs = np.asarray([tup[1] for tup in result_list_of_tuples])
    return stdevs.mean()



def accuracy_LR(X_train, X_validation, Y_train, Y_validation):
    lr = LogisticRegression(solver='liblinear', multi_class='ovr')
    lr.fit(X_train, Y_train)
    predictions = lr.predict(X_validation)
    return accuracy_score(Y_validation, predictions)#

def testset_validation_LogReg(train_validation_data):
    'make predictions on validation dataset'
    X_train, X_validation, Y_train, Y_validation = train_validation_data
    lr = LogisticRegression(solver='liblinear', multi_class='ovr')
    lr.fit(X_train, Y_train)
    predictions = lr.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

def all_validate(df_group1, df_group2, n):
        'returns mean, std, and all n results as nested list for repeated validations on test sample'
        results_list = []
        for i in range(n):
            sample = make_sample(df_group1, df_group2)
            array = sample.to_numpy()
            X = array[:, 0:(sample.shape[1] - 1)]
            Y = array[:, (sample.shape[1] - 1)]
            validation_size = 0.20
            seed = 7

            X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y,
                                                                                            test_size=validation_size,
                                                                                            random_state=seed)
            lr = LogisticRegression(solver='liblinear', multi_class='ovr')
            lr.fit(X_train, Y_train)
            predictions = lr.predict(X_validation)
            results_list.append(accuracy_score(Y_validation, predictions))
        return [np.asarray(results_list).mean(), np.asarray(results_list).std(), results_list]

def all_validate_mean(results_list):
        return np.asarray(results_list).mean()

def all_validate_std(results_list):
        'returns the std for repeated validations on test sample from functions all_validate and all_validate_random'
        return np.asarray(results_list).std()