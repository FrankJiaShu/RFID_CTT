import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from dataset.data_read import *
from fill.neighbour_filling import *
from filter.w_mean_filter import filter_mean
from filter.kalman_filter import filter_kalman


root = os.path.abspath(__file__)
sys.path.append('/'.join(root.split('/')[:-2])) 

font3 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 18,
}


root = os.path.dirname(__file__) + '/'
print(root)


def guass(x, sigma=1, mu=0):
    return math.e**(-(x-mu)**2 / 2 / sigma**2) / (2*math.pi)**0.5 / sigma


# print(k)


def filter_guass(arr):
    
    k = []
    for xi in [-2, -1, 0, 1, 2]:    
        gi = guass(xi)
        gi = round(gi, 4)
        k.append(gi)

    res = []
    for i in range(len(arr)):
        tmp = arr[i] * k[2]
        d = k[2]
        if i - 1 >=0:
            tmp += arr[i-1] * k[1]
            d += k[1]
        if i - 2 >= 0:
            tmp += arr[i-2] * k[0]
            d += k[0]
        if i + 1 < len(arr):
            tmp += arr[i+1] * k[3]
            d += k[3]
        if i + 2 < len(arr):
            tmp += arr[i+2] * k[4]
            d += k[4]
        res.append(tmp / d)
    return res


def plt_guass_filter(i=591):
    X, Y  = load_data()
    sx, xy = X[i], Y[i]
    print(xy)
    sx = neighbour_fill(sx)
    plt.plot(list(range(50)), sx, label='原信号')
    sx_filter_mean = filter_mean(sx)
    # print(sx_filter_mean)
    plt.plot(list(range(50)), sx_filter_mean, label='均值滤波', linestyle=":")
    
    sx_filter_guass = filter_guass(sx)
    # print(sx_filter_guass)
    plt.plot(list(range(50)), sx_filter_guass, label='高斯滤波', linestyle="--")
    
    sx_filter_kalman = filter_kalman(sx)
    # print(sx_filter_kalman)
    plt.plot(list(range(50)), sx_filter_kalman, label='卡尔曼滤波', linestyle="-.")
    plt.legend()
    plt.title("坐标：" + str((xy[0], xy[1])))
    plt.xlabel('Step', font3)
    plt.ylabel('RSSI / dBm', font3)
    plt.show()


if __name__ == '__main__':
    plt_guass_filter()
    