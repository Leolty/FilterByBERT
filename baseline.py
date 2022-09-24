# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 12:00:08 2022

@author: Leo
"""

from baseline_data_process import *
from Semi_EM_NB import Semi_EM_MultinomialNB
from sklearn import metrics
from performance_metrics import get_accuracy
from performance_metrics import get_f_measure
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
import joblib
import random
from configure import *



def Semi_EM_NB(appname: str):
    training_data_list = ["datasets/"+appname+"/trainL/info.txt",
                          "datasets/"+appname+"/trainL/non-info.txt"]
    training_data_info = "datasets/"+appname+"/trainL/info.txt"
    training_data_noninfo = "datasets/"+appname+"/trainL/non-info.txt"
    trainU_data = "datasets/"+appname+"/trainU/unlabeled.txt"

    # get training data
    mapping = extract_words_and_add_to_dict(training_data_list)

    training_data1 = get_data([training_data_info], mapping)
    training_data0 = get_data([training_data_noninfo], mapping)

    sample_num = SAMPLE_NUM
    random.seed(SEED_INFO)
    l1 = random.sample(range(1,len(training_data1)), sample_num)

    temp1 = []
    for i in l1:
        temp1.append(training_data1[i])
    temp1 = np.array(temp1)
    
    random.seed(SEED_NONINFO)
    l0 = random.sample(range(1,len(training_data0)), sample_num)

    temp0 = []
    for i in l0:
        temp0.append(training_data0[i])
    temp0 = np.array(temp0)


    trainY = np.ones(temp1.shape[0], dtype=int)
    trainY = np.append(trainY, np.zeros(temp0.shape[0], dtype=int))
    trainX = np.append(temp1, temp0, axis=0)

    random.seed(2018)
    if appname in ["facebook", "swift+facebook"]:
        trainU = np.array(random.sample(list(get_data([trainU_data], mapping)), 20000))
    else:
        trainU = get_data([trainU_data], mapping)

    # train model

    trainX, trainY = shuffle(trainX,trainY)
    
    clf = Semi_EM_MultinomialNB()
    clf.fit(trainX, trainY, trainU)

    # get test data
    test_data_info = "datasets/"+appname+"/test/info.txt"
    test_data_noninfo = "datasets/"+appname+"/test/non-info.txt"

    test_data0 = get_data([test_data_info], mapping)
    testY = np.ones(test_data0.shape[0], dtype=int)

    test_data1 = get_data([test_data_noninfo], mapping)
    testY = np.append(testY, np.zeros(test_data1.shape[0], dtype=int))

    testX = np.append(test_data0, test_data1, axis=0)

    testX, testY = shuffle(testX,testY)
    
    # predict with model
    result = clf.predict(testX)
    
    acc, f = get_accuracy(result, testY), get_f_measure(result,testY)

    return acc, f


def Baseline_Model(appname:str, clf):
    training_data_list = ["datasets/"+appname+"/trainL/info.txt",
                          "datasets/"+appname+"/trainL/non-info.txt"]
    training_data_info = "datasets/"+appname+"/trainL/info.txt"
    training_data_noninfo = "datasets/"+appname+"/trainL/non-info.txt"

    # get training data
    mapping = extract_words_and_add_to_dict(training_data_list)

    training_data1 = get_data([training_data_info], mapping)
    training_data0 = get_data([training_data_noninfo], mapping)

    sample_num = SAMPLE_NUM
    random.seed(SEED_INFO)
    l1 = random.sample(range(1,len(training_data1)), sample_num)

    temp1 = []
    for i in l1:
        temp1.append(training_data1[i])
    temp1 = np.array(temp1)
    
    random.seed(SEED_NONINFO)
    l0 = random.sample(range(1,len(training_data0)), sample_num)

    temp0 = []
    for i in l0:
        temp0.append(training_data0[i])
    temp0 = np.array(temp0)


    trainY = np.ones(temp1.shape[0], dtype=int)
    trainY = np.append(trainY, np.zeros(temp0.shape[0], dtype=int))
    trainX = np.append(temp1, temp0, axis=0)

    # train model

    trainX, trainY = shuffle(trainX,trainY)
    
    clf.fit(trainX, trainY)

    # get test data
    test_data_info = "datasets/"+appname+"/test/info.txt"
    test_data_noninfo = "datasets/"+appname+"/test/non-info.txt"

    test_data0 = get_data([test_data_info], mapping)
    testY = np.ones(test_data0.shape[0], dtype=int)

    test_data1 = get_data([test_data_noninfo], mapping)
    testY = np.append(testY, np.zeros(test_data1.shape[0], dtype=int))

    testX = np.append(test_data0, test_data1, axis=0)

    testX, testY = shuffle(testX,testY)
    
    # predict with model
    result = clf.predict(testX)
    
    acc, f = get_accuracy(result, testY), get_f_measure(result,testY)

    return acc, f
    