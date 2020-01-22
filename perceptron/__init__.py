#!/bin/python3

import numpy as np

from perceptron import PerceptronClassifier

x = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
y = np.array([[0],[1],[1],[0]])

# print(x)
# print(y)

pc = PerceptronClassifier(lr=1)
# print(pc)
pc.fit(x, y)