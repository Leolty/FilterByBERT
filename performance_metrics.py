# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:54:38 2022

@author: TianyangLiu
"""

# NOTE: precision and recall divide by 0 when arrays have no 1's

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from time import time
from copy import deepcopy
from sklearn import metrics

def get_accuracy(predictions, labels):
    assert ((predictions.shape == labels.shape) and (predictions.size != 0)), \
            "Arguments have unequal shapes"

    correct_predictions = np.count_nonzero(predictions == labels)
    total_predictions = predictions.shape[0]
    return correct_predictions/total_predictions

def get_precision(predictions, labels):
    assert ((predictions.shape == labels.shape) and (predictions.size != 0)), \
            "Arguments have unequal shapes"

    true_pos = np.count_nonzero(np.logical_and(predictions == 1, labels == 1))
    false_pos = np.count_nonzero(np.logical_and(predictions == 1, labels == 0))
    return true_pos / (true_pos + false_pos)

def get_recall(predictions, labels):
    assert ((predictions.shape == labels.shape) and (predictions.size != 0)), \
            "Arguments have unequal shapes"

    true_pos = np.count_nonzero(np.logical_and(predictions == 1, labels == 1))
    false_neg = np.count_nonzero(np.logical_and(predictions == 0, labels == 1))
    return true_pos / (true_pos + false_neg)

def get_f_measure(predictions, labels):
    assert ((predictions.shape == labels.shape) and (predictions.size != 0)), \
            "Arguments have unequal shapes"

    precision = get_precision(predictions, labels)
    recall = get_recall(predictions, labels)
    return (2 * precision * recall) / (precision + recall)

def cross_validation(clf, data_X, data_y, unlabeled, n_folds=10):
    print('=' * 80)
    print("Validation: ")
    kf = StratifiedKFold(n_splits=n_folds)
    start_time = time()
    train_scores = list() # training accuracy
    fold_count = 1
    original_clf = deepcopy(clf)
    avg_accuracy = 0
    avg_f1 = 0
    print(original_clf)
    for train_ids, valid_ids in kf.split(data_X, data_y):
        cv_clf = deepcopy(original_clf)
        print("Fold # %d" % fold_count)
        fold_count += 1
        train_X, train_y, valid_X, valid_y = data_X[train_ids], data_y[train_ids], data_X[valid_ids], data_y[valid_ids]
#        if all(unlabeled==None):
#            print("No unlabeled")
#            cv_clf.fit(train_X, train_y)
#        else:
        print("refine")
        cv_clf.fit(train_X, train_y, unlabeled)
        cv_clf.partial_fit(train_X, train_y, unlabeled)
        pred = cv_clf.predict(valid_X)
        scores = dict()
        scores['accuracy'] = metrics.accuracy_score(valid_y, pred)
        scores['recall'] = metrics.recall_score(valid_y, pred, average='macro')
        scores['precision'] = metrics.precision_score(valid_y, pred, average='macro')
        scores['f1_score'] = metrics.f1_score(valid_y, pred, average='macro')
        train_scores.append(scores)
        avg_accuracy += scores['accuracy']
        avg_f1 += scores['f1_score']
    train_time = time() - start_time
    print("Validation time: %0.3f seconds" % train_time)
    print("Average training accuracy: %0.3f" % (avg_accuracy/n_folds))
    print("Average F1 meausre:%0.3f" % (avg_f1/n_folds))
    return train_scores, train_time
