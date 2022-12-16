import sys, os
root = os.path.abspath(__file__)
sys.path.append('/'.join(root.split('/')[:-2]))
sys.path.append('/'.join(root.split('/')[:-2]) + '/' + 'filter')

from sklearn import svm
from sklearn.metrics import mean_absolute_error
from filter.data_comb import *
import pickle

X_train, Y_train = np.array(X_train_guass), Y_train_0
X_test, Y_test = np.array(X_test_guass), Y_test_0


Y_train_x = Y_train[:, 0]
Y_train_y = Y_train[:, 1]
Y_test_x = Y_test[:, 0]
Y_test_y = Y_test[:, 1]
print(X_train.shape, Y_train_x.shape)

# 预测x坐标
regr_x = svm.SVR(kernel='poly', degree=2)
# regr_x = svm.SVR()
regr_x.fit(X_train, Y_train_x)
print('ok1')
# 预测y坐标
regr_y = svm.SVR(kernel='poly', degree=2)
# regr_y = svm.SVR()
regr_y.fit(X_train, Y_train_y)
print('ok2')

px = regr_x.predict(X_test)
py = regr_y.predict(X_test)


root = os.path.abspath(__file__)
path = '/'.join(root.split('/')[:-1]) + '/SAVED_MODEL/'
print(path)
with open(path + 'SVM_X_2022_2.pkl', 'wb') as fout:
    pickle.dump(regr_x, fout)
with open(path + 'SVM_Y_2022_2.pkl', 'wb') as fout:
    pickle.dump(regr_y, fout)

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