import numpy as np
import random
import math
import copy
import pandas as pd

import os

root = os.path.dirname(__file__) + '/'


random.seed(2022)
# 阅读器的位置, 随机100个采集数据点
reader_pos = [[5, float('%.2f' % (random.random() * 5))] for i in range(50)]
reader_pos = sorted(reader_pos, key=lambda x: x[1])
print(reader_pos)
# exit(0)
# 数据集生成，2200条，2000条作为训练集，200条作为测试集
random.seed(20221)
x1 = [float('%.2f' % (random.uniform(0, 5))) for i in range(2000)]
y1 = [float('%.2f' % (random.uniform(0, 5))) for i in range(2000)]
random.seed(20222)
x2 = [float('%.2f' % (random.uniform(0, 5))) for i in range(200)]
y2 = [float('%.2f' % (random.uniform(0, 5))) for i in range(200)]
x = x1 + x2
y = y1 + y2
lable = [[xi, yi] for xi, yi in zip(x, y)]


# 计算欧式距离
def eu_distance(p1, p2):
    a = np.array(p1)
    b = np.array(p2)
    # 加入噪声，误差，-1 ～ 1 标准高斯分布
    noise = random.gauss(0, 1)

    return min(5, max(0.01, np.linalg.norm(a - b) + noise))


# 仿真计算信号强度
def dis_to_rss(d, d0=0.5, pl0=-47):
    if random.random() < 0.02: return 0
    x_th = random.gauss(0, 1)
    n = -2.55
    rss = pl0 + 10 * n * math.log10(d / d0) + x_th
    rss = max(-90, min(-20, rss))
    return '%.0f' % (rss)


def is_blocked(li, ri, block):
    rx, ry = block
    a1, b1 = int(rx / 0.5) * 0.5, int(ry / 0.5) * 0.5
    a2, b2 = a1 + 0.5, b1 + 0.5 
    
    x1, y1 = li
    x2, y2 = ri
    def fun01(x):
        return (y2 - y1) / (x2 - x1 + 1e-7) * (x -x1)  + y1
    
    if a1 >= x1 and b1<= fun01(a1)<=b2 or b1<= fun01(a2)<=b2:
        return True 
    
    return False
    

data = []
rssi_list = []
for li in lable:
    t = []
    block = (random.uniform(0, 5), random.uniform(0, 5))
    for ri in reader_pos:
        di = eu_distance(li, ri)
        rssi = dis_to_rss(di)
        if is_blocked(li, ri, block) and random.random()<0.5: 
            rssi = 0
        t.append(rssi)
        rssi_list.append(float(rssi))
    t.extend(li)
    data.append(copy.copy(t))
    
print(max(rssi_list), min(rssi_list))

print(np.shape(data))
df = pd.DataFrame(data[:2000], index=None, columns=None)
df.to_csv(root + 'location_data_train.csv', index=None)
df = pd.DataFrame(data[2000:], index=None, columns=None)
df.to_csv(root + 'location_data_test.csv', index=None)


