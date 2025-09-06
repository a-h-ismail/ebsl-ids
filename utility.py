#!/usr/bin/env python3
import pickle
from sklearn.feature_selection import SelectKBest, f_classif

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


def get_metrics(expected, prediction):
    # Can be optimized by calculating accuracy, precision, recall... directly from the confusion matrix
    print("Metrics:")
    print('Accuracy:   %.3g' % accuracy_score(expected, prediction))
    print('Precision:  %.3g' % precision_score(expected, prediction))
    print('Recall:     %.3g' % recall_score(expected, prediction))
    print('F1 score:   %.3g' % f1_score(expected, prediction))
    print('ROC-AUC:    %.3g' % roc_auc_score(expected, prediction))
    print('MCC score:  %.3g' % matthews_corrcoef(expected, prediction))
    print('\nConfusion Matrix :\n', confusion_matrix(expected, prediction))
    print()


def from_csv(filename, label_name: str, fields_to_drop: tuple | list):
    """Reads the dataset from CSV file, then splits the labels from the data, 
    drops the specified fields and replaces missing values with the mean of column"""
    actual_data = pd.read_csv(filename)
    # Drop unwanted fields
    actual_data.drop(labels=fields_to_drop, axis=1, inplace=True)
    # Convert strings to numbers (assumes all features are numerical)
    actual_data[label_name] = pd.to_numeric(actual_data[label_name], errors='coerce')

    # Drop any unlabeled features
    actual_data.dropna(subset=[label_name], inplace=True)

    labels = actual_data[label_name]

    actual_data.drop([label_name], axis=1, inplace=True)

    for col_name in actual_data.columns:
        # Convert any string to its numeric representation
        actual_data[col_name] = pd.to_numeric(actual_data[col_name], errors='coerce')
        # Any missing value is replaced with the mean of the column
        mean = actual_data[col_name].mean()
        actual_data.replace({col_name: np.nan}, mean, inplace=True)

    return actual_data, labels


def select_k_best(features: pd.DataFrame, labels: pd.DataFrame, k=-1):
    """Selects the K best features based on the ANOVA metric"""
    if k != -1:
        selector = SelectKBest(f_classif, k=k)
        selector.fit(features, labels)
        cols_idxs = selector.get_support(indices=True)
        features = features.iloc[:, cols_idxs]
        print("Selected features:", features.columns)

    return features


def fit_and_scale_datasets(training_set: pd.DataFrame, testing_set: pd.DataFrame, scaler):
    """Fits the scaler to the training set. Returns the scaled train/test sets and the scaler object"""
    scaler.set_output(transform='pandas')
    scaler.fit(training_set)
    training_set = pd.DataFrame(scaler.transform(training_set))
    testing_set = pd.DataFrame(scaler.transform(testing_set))
    return training_set, testing_set, scaler


def store_model(name, model, scaler):
    """Store the model and the training set scaler"""
    model_file = open(name+"_model", "wb")
    pickle.dump((model, scaler), model_file)
    model_file.close()


def load_model(name):
    """Retrieves model and scaler from the saved file"""
    model_file = open(name+"_model", "rb")
    model, scaler = pickle.load(model_file)
    return model, scaler
