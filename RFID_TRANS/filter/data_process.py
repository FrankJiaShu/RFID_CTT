
from w_mean_filter import *
from rssi_guass_filter import *
from kalman_filter import *
import imp
import sys, os
root = os.path.abspath(__file__)
sys.path.append('/'.join(root.split('/')[:-1])) 

X_train, Y_train = load_data()
print(X_train.shape)
X_test, Y_test = load_data('process')
print(X_test.shape)

X_test_nf = [neighbour_fill(xi) for xi in X_test]
X_train_nf = [neighbour_fill(xi) for xi in X_train]


X_train_wmf = [filter_mean(xi) for xi in X_train_nf]
X_test_wmf = [filter_mean(xi) for xi in X_test_nf]


X_train_rgf = [filter_guass(xi) for xi in X_train_nf]
X_test_rgf = [filter_guass(xi) for xi in X_test_nf]

X_train_kf = [filter_kalman(xi) for xi in X_train_nf]
X_test_kf = [filter_kalman(xi) for xi in X_test_nf]



