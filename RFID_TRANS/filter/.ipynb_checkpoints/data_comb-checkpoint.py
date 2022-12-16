# 各种组合
import sys, os
root = os.path.abspath(__file__)
sys.path.append('/'.join(root.split('/')[:-1])) 
sys.path.append('/'.join(root.split('/')[:-2])) 
from fill.neighbour_filling import neighbour_fill
from fill.hot_filling import hot_fill
from filter.w_mean_filter import *
from filter.rssi_guass_filter import filter_guass
from filter.kalman_filter import filter_kalman
import pickle

# 0填充
X_train_0, Y_train_0  = load_data()
X_test_0, Y_test_0  = load_data('process')

X_train_neighbour = [neighbour_fill(xi) for xi in X_train_0]
X_test_neighbour = [neighbour_fill(xi) for xi in X_test_0]

print(np.shape(X_train_neighbour))

# X_train_hot = [hot_fill(xi) for xi in X_train_0]
# X_test_hot = [hot_fill(xi) for xi in X_test_0]

# with open('X_train_hot.pkl', 'wb') as fout:
#     pickle.dump(X_train_hot, fout)
# with open('X_test_hot.pkl', 'wb') as fout:
#     pickle.dump(X_test_hot, fout)

# with open('X_train_hot.pkl', 'rb') as fin:
#     X_train_hot = pickle.load(fin)
# with open('X_test_hot.pkl', 'rb') as fin:
#     X_test_hot = pickle.load(fin)

# print(np.shape(X_train_hot))
# print(np.shape(X_test_hot))


X_train_guass = [filter_guass(xi) for xi in X_train_neighbour]
X_test_guass = [filter_guass(xi) for xi in X_test_neighbour]

