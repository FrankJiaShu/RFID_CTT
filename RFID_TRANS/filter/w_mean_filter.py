import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset.data_read import *
from fill.neighbour_filling import *
root = os.path.abspath(__file__)
sys.path.append('/'.join(root.split('/')[:-2]))


def filter_mean(arr):
    wights = [1, 1, 1, 1, 4]
    res = [arr[0]]
    for i in range(1, len(arr)):
        tmp = arr[i] * 4
        k = 4
        for j in range(1, 5):
            ind = i - j
            if ind < 0:
                break
            tmp += arr[ind]
            k += 1
        res.append(tmp / k)
    return res

# arr = [1, 2, 3, 4, 5, 6]
# print(filter_mean(arr))


def plt_mean_filter(i=906):
    X, Y  = load_data()
    sx, xy = X[i], Y[i]
    sx = neighbour_fill(sx)
    print(sx)
    sx_filter = filter_mean(sx)
    print(sx_filter)
    plt.plot(list(range(50)), sx, label='原信号')
    plt.plot(list(range(50)), sx_filter, label='均值滤波')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X, Y = load_data()
    sx, xy = X[906], Y[906]
    sx = neighbour_fill(sx)
    print(sx)
    sx_filter = filter_mean(sx)
    print(sx_filter)
    # plt_mean_filter()