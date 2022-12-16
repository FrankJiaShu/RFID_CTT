import sys, os
root = os.path.abspath(__file__)
sys.path.append('/'.join(root.split('/')[:-2])) 

import numpy as np
import math
from pprint import pprint as prt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, explained_variance_score, r2_score, mean_squared_error
from DATA.data_read import *

import time


class my_GRNN:
    def __init__(self, trainX, trainY, sigma=1.0):
        self.trainX = trainX
        self.trainY = trainY
        self.sigma = sigma

    # 两点距离
    def distance(self, X, Y):
        return np.linalg.norm(X - Y)

    # 测试样本(一条)与所有训练样本的距离
    def distance_mat(self, testX):
        Eu_dis = []
        for Xi in self.trainX:
            Eu_dis.append(self.distance(testX, Xi))
        return Eu_dis

    # 模式层
    def Gauss(self, Eu_dis):
        g_list = []
        for ei in Eu_dis:
            gi = math.exp(-ei / (2 * (self.sigma ** 2)))
            g_list.append(gi)
        return g_list

    # 求和层
    def sum_layer(self, g_list):
        s_list = []
        s0 = sum(g_list)
        s_list.append(s0)

        g_list = np.array(g_list)
        m, n = np.shape(self.trainY)
        for i in range(n):
            s_list.append(np.sum(g_list * self.trainY[:, i]))
        return s_list

    # 输出层
    def output_layer(self, s_list):
        ans = []
        for i in range(1, len(s_list)):
            if s_list[0] == 0:
                ans.append(0)
                continue
            ans.append(s_list[i] / s_list[0])
        return ans

    def GRNN_main(self, testX):
        Eu_dis = self.distance_mat(testX)
        g_list = self.Gauss(Eu_dis)
        s_list = self.sum_layer(g_list)
        predict = self.output_layer(s_list)
        return predict


if __name__ == '__main__':
    # X_train, Y_train = load_data('train')
    # X_test, Y_test = load_data('process')
    from data_process import *
    X_train, Y_train = np.array(X_train_kf), Y_train
    X_test, Y_test = np.array(X_test_kf), Y_test
    # print(X_train.shape)
    
    myGRNN = my_GRNN(X_train, Y_train, sigma=0.3)
    a = myGRNN.GRNN_main(X_test[5])
    print(a, Y_test[5])

    begin_time = time.time()
    pxy = []
    for xt in X_test:
        pxy.append(myGRNN.GRNN_main(xt))
        
    end_time = time.time()
    print('耗时：', end_time - begin_time)
    
    print(np.shape(pxy))

    
    print('mae = ', mean_absolute_error(Y_test, pxy))
    print('mse = ', mean_squared_error(Y_test, pxy))
    print('rmse = ', mean_squared_error(Y_test, pxy)**0.5)
    print('evs = ', explained_variance_score(Y_test, pxy))
    print('r2 = ', r2_score(Y_test, pxy))


    mmax = 0
    mmin = 10000
    for i in range(len(pxy)):
        mmax = max(mean_absolute_error(Y_test[i], pxy[i]), mmax)
        mmin = min(mean_absolute_error(Y_test[i], pxy[i]), mmin)
    print(mmax, mmin)


    