import sys
import os


root = os.path.abspath(__file__)
sys.path.append('/'.join(root.split('/')[:-2])) 

import numpy as np
import matplotlib.pyplot as plt
from dataset.data_read import *

# mac用这个解决中文显示问题
plt.figure()
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams.update({'font.size': 18})

font1 = {
    'family': 'Songti SC',
    'weight': 'normal',
    'size': 20,
}
font2 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 20,
}


def fun_dst(arr1, arr2):
    inds = [i for i in range(len(arr1)) if arr1[i] != 0 and arr2[i] != 0]
    arr1 = np.array(arr1)[inds]
    arr2 = np.array(arr2)[inds]
    return np.linalg.norm(arr1 - arr2)


def hot_fill(arr):
    arr_0 = [i for i in range(len(arr)) if arr[i] == 0]
    # print(arr)
    # print(arr_0)
    X, Y = load_data()
    closest, dst = [], 1000000
    for xi in X:
        if not np.all(xi[arr_0]): continue
        if fun_dst(xi, arr) < dst:
            closest = xi
            dst = fun_dst(xi, arr)     
    # print(closest)
    # print(arr)
    res = [arr[i] if arr[i]!=0 else closest[i] for i in range(len(arr))]
    # print(res)
    return res


# 热卡填充绘图
def plt_hot_fill(X, Y):
    x = list(range(50)) 
    y = X
    y = hot_fill(y)
    # plt.title("热卡填充", font1)
    plt.plot(x, y)
    plt.xlabel('Step', font2)
    plt.ylabel('RSSI / dBm', font2)
    miss_X = [i for i in range(len(X)) if X[i]==0]
    miss_Y = [y[i] for i in miss_X]
    plt.scatter(miss_X, miss_Y, c='r')
    plt.show()


if __name__ == '__main__':
    X, Y = load_data()
    plt_hot_fill(X[591], Y[591])
    
          