import numpy as np
import pandas as pd
import torch
from data_reader import read_review_data
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures


def main():
    app_name = "swiftkey"
    training_data_info = "datasets/" + app_name + "/trainL/info.txt"
    training_data_noninfo = "datasets/" + app_name + "/trainL/non-info.txt"
    # read data
    training_data1 = read_review_data([training_data_info])
    training_data0 = read_review_data([training_data_noninfo])

    trainY = np.ones(training_data1.shape[0], dtype=int)
    trainY = np.append(trainY, np.zeros(training_data0.shape[0], dtype=int))

    trainX = np.append(training_data0, training_data1, axis=0)

    print(len(trainX), len(trainY))


if __name__ == '__main__':
    main()
