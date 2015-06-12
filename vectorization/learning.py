import sys
import math
import random
import numpy

from sklearn import grid_search
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#############################################################################
# linear svm for 10 times
# Input:
#     train_data
#     test_data
#     num_fold = number of fold for cross validation
# Output:
#     train_acc
#     valid_acc
#     test_acc
#     c
#############################################################################
def logistic_regression(train_data, test_data, num_fold):

    best_valid_ACC = 0
    # Tune parameters
    for z in range(-8, 8):
        C = pow(2, z)
        result_train = list()
        result_valid = list()
        # Random for 10 times
        for i in range(1):
            # Random permutation
            train_data = train_data[random.sample(range(train_data.shape[0]), train_data.shape[0]), :]
                
            # Do cross-validation
            skf = cross_validation.StratifiedKFold(train_data[:, 0], n_folds=num_fold)        
            clf = LogisticRegression(C=C)
            for train_index, valid_index in skf:
                clf.fit(train_data[train_index, 1:], train_data[train_index, 0])
                train_pred = clf.predict_proba(train_data[train_index, 1:])
                valid_pred = clf.predict_proba(train_data[valid_index, 1:])
                # train_acc = accuracy_score(train_data[train_index, 0], train_pred)
                # valid_acc = accuracy_score(train_data[valid_index, 0], valid_pred)
                train_acc = roc_auc_score(train_data[train_index, 0], train_pred[:, 1])
                valid_acc = roc_auc_score(train_data[valid_index, 0], valid_pred[:, 1])
                result_train.append(train_acc)
                result_valid.append(valid_acc)
        
        # If mean accuracy greater than best accuracy, then record it
        if sum(result_valid)/(1.0*num_fold) > best_valid_ACC:
            best_valid_ACC = sum(result_valid)/(1.0*num_fold)
            best_train_ACC = sum(result_train)/(1.0*num_fold)
            best_C = C

    # Predict test data with best c
    clf = LogisticRegression(C=best_C)
    clf.fit(train_data[:, 1:], train_data[:, 0])
    test_pred = clf.predict_proba(test_data[:, 1:])
    test_ACC = roc_auc_score(test_data[:, 0], test_pred[:, 1])
    
    return best_train_ACC, best_valid_ACC, test_ACC, best_C
    
    
    
#############################################################################
# Intersection Kernel
# Input:
#     a
#     b
# Output:
#     K
#############################################################################    
def inter_kernel(a, b):
    row_a = a.shape[0]
    row_b = b.shape[0]
    K = numpy.zeros([row_a, row_b])
    
    for i in range(row_a):
        for j in range(row_b):
            K[i, j] = numpy.sum(numpy.minimum(a[i, :], b[j, :]))
    return K
    
    
#############################################################################
# inter svm for 10 times
# Input:
#     train_data
#     test_data
#     num_fold = number of fold for cross validation
# Output:
#     train_acc
#     valid_acc
#     test_acc
#     model_params: c
#############################################################################
def inter_svm(train_data, test_data, num_fold):
    tmp = inter_kernel(test_data[:, 1:], train_data[:, 1:])
    test_data = numpy.hstack([test_data[:, :1], tmp])
    tmp = inter_kernel(train_data[:, 1:], train_data[:, 1:])
    train_data = numpy.hstack([train_data[:, :1], tmp])

    best_valid_ACC = 0
    # Tune parameters
    for z in range(-8, 8):
        C = pow(2, z)
        result_train = list()
        result_valid = list()

        # Do cross-validation
        skf = cross_validation.StratifiedKFold(
                                   train_data[:, 0], 
                                   n_folds=num_fold
                               )        
        clf = SVC(C, kernel='precomputed', probability=True)
        for train_index, valid_index in skf:
            clf.fit(train_data[train_index, :][:, train_index+1], train_data[train_index, 0])
            train_pred = clf.predict_proba(train_data[train_index, :][:, train_index+1])
            valid_pred = clf.predict_proba(train_data[valid_index, :][:, train_index+1])
            train_acc = roc_auc_score(train_data[train_index, 0], train_pred[:, 1])
            valid_acc = roc_auc_score(train_data[valid_index, 0], valid_pred[:, 1])
            #train_pred = clf.predict(train_data[train_index, :][:, train_index+1])
            #valid_pred = clf.predict(train_data[valid_index, :][:, train_index+1])

            #train_acc = accuracy_score(train_data[train_index, 0], train_pred)
            #valid_acc = accuracy_score(train_data[valid_index, 0], valid_pred)
            result_train.append(train_acc)
            result_valid.append(valid_acc)
        # If mean accuracy greater than best accuracy, then record it
        if sum(result_valid)/num_fold > best_valid_ACC:
            best_valid_ACC = sum(result_valid)/num_fold
            best_train_ACC = sum(result_train)/num_fold
            best_C = C

    # Predict test data with best C
    clf = SVC(best_C, kernel='precomputed', probability=True)
    clf.fit(train_data[:, 1:], train_data[:, 0])
    test_pred = clf.predict_proba(test_data[:, 1:])
    test_ACC = roc_auc_score(test_data[:, 0], test_pred[:, 1])
    
    return best_train_ACC, best_valid_ACC, test_ACC, best_C