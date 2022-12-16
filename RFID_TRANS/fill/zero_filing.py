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

X, Y = load_data()
print(np.shape(X), np.shape(Y))

print(Y[591])
print(X[591])


# 零值填充绘图
def plt_zero_filling(X, Y):
    x = list(range(50))
    y = X
    # plt.title("零值填充", font1)
    plt.plot(x, y)
    plt.xlabel('Step', font2)
    plt.ylabel('RSSI / dBm', font2)
    miss_X = [i for i in range(len(X)) if X[i] == 0]
    miss_Y = [0 for i in miss_X]
    plt.scatter(miss_X, miss_Y, c='r')
    plt.show()


if __name__ == '__main':
    plt_zero_filling(X[591], Y[591])
