import torch
import sys
import os
from MODEL.NN_LOC import LOCATION_NN
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_error
# from sklearn.externals import joblibxs
# from MODEL.train_model_step import evaluate_step
from sklearn.metrics import mean_absolute_error, explained_variance_score, r2_score, mean_squared_error
from filter.data_comb import *

root = os.path.abspath(__file__)
sys.path.append('/'.join(root.split('/')[:-3]))
sys.path.append('/'.join(root.split('/')[:-3]) + '/' + 'filter')
print('/'.join(root.split('/')[:-3])) 
sys.path.append('/'.join(root.split('/')[:-2])) 

X_train, Y_train = np.array(X_train_guass), Y_train_0
X_test, Y_test = np.array(X_test_guass), Y_test_0

Y_train_x = Y_train[:, 0]
Y_train_y = Y_train[:, 1]
Y_test_x = Y_test[:, 0]
Y_test_y = Y_test[:, 1]
print(X_train.shape, Y_train_x.shape)

# 预测x坐标
# regr_x = svm.SVR(kernel='poly', degree=2)
regr_x = svm.SVR()
regr_x.fit(X_train, Y_train_x)
print('ok1')
# 预测y坐标
regr_y = svm.SVR(kernel='poly', degree=2)
# regr_y = svm.SVR()
regr_y.fit(X_train, Y_train_y)
print('ok2')

px = regr_x.predict(X_test)
py = regr_y.predict(X_test)

# joblib.dump(regr_x, 'SVR_x_2020.model')
# joblib.dump(regr_y, 'SVR_y_2020.model')


print(px[:10])
print(Y_test_x[:10])


_px = px[:, np.newaxis]
_py = py[:, np.newaxis]
pxy = np.hstack((_px, _py))
print(pxy[:20])
print(Y_test[:20])
print(np.shape(pxy))
print(np.shape(Y_test))

print(mean_absolute_error(Y_test, pxy))