#!/bin/python3

import numpy as np
from mlp import MLPClassifier

def basic():
    print("-------------basic---------------")

    x = np.array([[0,0],[0,1], [1, 1]])
    y = np.array([[1],[0], [0]])
    # x = np.array([[0, 0]])
    # y = np.array([[1]])

    # print(x)
    # print(y)
    pc = None
    try:
        pc = MLPClassifier([2], 1)
        # print(pc)
        print(pc.fit(x, y, standard_weight=1))
        print("fake score", pc.score(x, y))
        # print(pc.fit(x, y).score(np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,0]]), np.array([[0],[0],[1],[1]])))
        # print(pc)
    except Exception as e:
        print("AN EXCEPTION OCCURRED!!!---------\n\r"*2, pc)
        raise e

basic()
