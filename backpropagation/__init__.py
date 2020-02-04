#!/bin/python3

import numpy as np
from mlp import MLPClassifier

def basic():
    print("-------------basic---------------")

    x = np.array([[0,0],[0,1], [1, 1]])
    y = np.array([[1],[0], [0]])

    # print(x)
    # print(y)

    pc = MLPClassifier([2])
    # print(pc)
    print(pc.fit(x, y, standard_weight=1))
    # print(pc.fit(x, y).score(np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,0]]), np.array([[0],[0],[1],[1]])))
    # print(pc)

basic()
