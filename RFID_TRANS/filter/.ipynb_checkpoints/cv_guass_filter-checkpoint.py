import cv2
import numpy as np
import os
import math
from pprint import pprint as prt


root = os.path.dirname(__file__) + '/'
print(root)
cat = cv2.imread(root + 'cat.png')
print(cat)
print(np.shape(cat))


def guass(x, sigma=1, mu=0):
    return math.e**(-(x-mu)**2 / 2 / sigma**2) / (2*math.pi)**0.5 / sigma

k = []

for x in [-1, 0, 1]:
    tmp = []
    for y in [-1, 0, 1]:
        tmp.append(guass(x) * guass(y))
    k.append(tmp)
# print(k)
k = np.array(k)
k = k / np.sum(k)
print(k)
        

dst = cv2.filter2D(src=cat, ddepth=-1, kernel=k)
cv2.imwrite(root + 'cat2.png', dst)
