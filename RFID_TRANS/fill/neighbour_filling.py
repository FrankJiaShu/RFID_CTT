import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset.data_read import *


root = os.path.abspath(__file__)
sys.path.append('/'.join(root.split('/')[:-2]))
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


def neighbour_fill(arr):
    res = []
    for i in range(len(arr)):
        if arr[i] != 0:
            res.append(arr[i])
        else:
            left_j, right_j = i, i
            while left_j >= 0:
                if arr[left_j] != 0:
                    break
                left_j -= 1
            while right_j < len(arr):
                if arr[right_j] != 0:
                    break
                right_j += 1
            if left_j == -1:
                left_j = -1000
            if right_j == len(arr):
                right_j = 1000
            if abs(i - left_j) < abs(i - right_j):
                res.append(arr[left_j])
            elif abs(i - left_j) == abs(i - right_j):
                res.append((arr[left_j] + arr[right_j]) / 2)
            else:
                res.append(arr[right_j])
    return res


# 近邻填充绘图
def plt_neighbour_fill(X, Y):
    x = list(range(50)) 
    y = X
    y = neighbour_fill(y)
    # plt.title("近邻填充", font1)
    plt.plot(x, y)
    plt.xlabel('Step', font2)
    plt.ylabel('RSSI / dBm', font2)
    miss_X = [i for i in range(len(X)) if X[i]==0]
    miss_Y = [y[i] for i in miss_X]
    plt.scatter(miss_X, miss_Y, c='r')
    plt.show()


if __name__ == '__main__':
    X, Y = load_data()
    plt_neighbour_fill(X[591], Y[591])
          