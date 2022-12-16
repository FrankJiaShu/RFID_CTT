import pandas as pd
import os
import numpy as np
from pprint import pprint as prt

root = os.path.dirname(__file__) + '/'


def load_data(mode='train'):
    path = root + 'location_data_train.csv' if mode == 'train' else root + 'location_data_test.csv'
    data = pd.read_csv(path)
    data_val = data.values
    X = data_val[:, :-2]
    Y = data_val[:, -2:]
    return X, Y


def load_data_prefix(mode='train'):
    X, Y = load_data(mode)
    new_X = []
    for xi in X:
        for j in range(1, len(xi)):
            if xi[j] == 0:
                xi[j] = xi[j-1]
        new_X.append(xi)
    new_X = np.array(new_X)
    return new_X, Y


def load_data_sufix(mode='train'):
    X, Y = load_data(mode)
    new_X = []
    for xi in X:
        for j in range(len(xi)-2, -1, -1):
            if xi[j] == 0:
                xi[j] = xi[j+1]
        new_X.append(xi)
    new_X = np.array(new_X)
    return new_X, Y


def load_data_ave(mode='train'):
    X, Y = load_data(mode)
    ave = np.average(X)
    new_X = []
    for xi in X:
        for j in range(len(xi)):
            if xi[j] == 0:
                xi[j] = ave
        new_X.append(xi)
    new_X = np.array(new_X)
    return new_X, Y

# load_data_prefix()
# X, Y = load_data_sufix()
# prt(X[1689])