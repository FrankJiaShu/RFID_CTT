from gc import collect
import math
from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt

collect_data = [(0.13, -32), (0.15, -35), (0.19, -40), 
                (0.3, -45), (0.5, -46), (0.55, -47), 
                (0.62, -48), (0.64, -49), (0.65, -50), 
                (0.68, -51), (0.7, -52), (0.76, -53),
                (0.82, -54), (0.86, -55), (0.91, -56),
                (1.18, -57), (1.3, -58), (1.4, -59),
                (1.5, -60), (1.7, -61), (1.9, -62),
                (2.2, -63), (2.3, -64), (2.5, -65)
]

x = [ci[0] for ci in collect_data]
y = [ci[1] for ci in collect_data]


d0 = 0.5
p0 = -46
def fun01(n,d):
    return p0 + 10 * n * math.log10(d / d0) 

def loss(y, py):
    res = 0
    for i in range(len(y)):
        res += (y[i] - py[i]) ** 2
    return res

L = []
nl = np.linspace(-10, 10, 2000)
bl = np.linspace(-2, 2, 100)
cl = []
for n in nl:
    py = [fun01(n, di) for di in x]
    L.append(loss(y, py))
    cl.append([n])

# plt.plot(nl, L)
# plt.show()

ind = int(np.argmin(L))
# print(ind)
# print(np.shape(cl))
print(L[ind], cl[ind])
x2 = np.linspace(0.01, 3, 40)
py = [fun01(cl[ind][0], xi) for xi in x2]
plt.plot(x, y, marker='^')
plt.plot(x2, py)
plt.show()

others = []

for yi in range(-57, -71, -1):
    mmin = 10000
    min_d = 10000
    for d_ans in np.linspace(0.9, 5, 411):
        yd = fun01(cl[ind][0],cl[ind][1],d_ans)
        if abs(yd - yi) < mmin:
            mmin = abs(yd - yi)
            min_d = d_ans
    others.append([min_d, yi])
print(others)
        


