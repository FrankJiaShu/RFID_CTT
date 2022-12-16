import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import random

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


root = os.path.dirname(__file__) + '/'

train_data = pd.read_csv(root + 'location_data_train.csv')
train_data_val = train_data.values

def plt_sample(i):
    x = list(range(50)) 
    y = train_data_val[i][:50]
    y = [yi if yi != 0 else None for yi in y]
    # plt.title("坐标: " + str((train_data_val[i][50], train_data_val[i][51])), font=font1)    
    plt.plot(x, y)
    plt.xlabel('Step', font2)
    plt.ylabel('RSSI / dBm', font2)
    

if __name__ == '__main__':
    random.seed(2022)
    rs = random.sample(list(range(2000)), 4)
    print(rs)
    

    # for i, ri in enumerate(rs):
    #     plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.5,hspace=0.5)
    #     plt.subplot(2, 2, i+1)
    #     plt_sample(ri)
    # plt.show()
    
    plt_sample(591)
    plt.show()
    
    
    



