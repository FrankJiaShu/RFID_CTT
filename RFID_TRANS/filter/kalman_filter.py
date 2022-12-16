import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset.data_read import *
from fill.neighbour_filling import *


root = os.path.abspath(__file__)
sys.path.append('/'.join(root.split('/')[:-2]))


def filter_kalman(arr):
    res = [arr[0]]
    e_mea = 2
    e_est = 5
    for i in range(1, len(arr)):
        arr_i = arr[i]
        kal = e_est / (e_est + e_mea)
        xi = res[-1] + kal * (arr_i - res[-1])
        res.append(xi)
        e_est = (1 - kal) * e_est
    return res

# arr = [1, 2, 3, 4, 5, 6]
# print(filter_mean(arr))


def plt_kalman_filter(i=1000):
    X, Y = load_data()
    sx, xy = X[i], Y[i]
    print(xy)
    sx = neighbour_fill(sx)
    # print(sx)
    sx_filter = filter_kalman(sx)
    # print(sx_filter)
    plt.plot(list(range(50)), sx, label='原信号')
    plt.plot(list(range(50)), sx_filter, label='卡尔曼滤波')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X, Y = load_data()
    sx, xy = X[906], Y[906]
    # print(xy)
    sx = neighbour_fill(sx)
    print(sx)
    sx_filter = filter_kalman(sx)
    print(sx_filter)
    plt_kalman_filter()