#!/bin/python3

import numpy as np

from perceptron import PerceptronClassifier
from arff import Arff

x = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
y = np.array([[0],[1],[1],[0]])

# print(x)
# print(y)

pc = PerceptronClassifier(lr=1, printIt=True)
# print(pc)
print(pc.fit(x, y, epochs=4).score(np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,0]]), np.array([[0],[0],[1],[1]])))

print("------------arff-------------------")

mat = Arff("../datasets/lineralyseparable.arff",label_count=1)
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)
PClass = PerceptronClassifier(lr=0.1,shuffle=False,deterministic=10)
PClass.fit(data,labels)
Accuracy = PClass.score(data,labels)
print("Accuray = [{:.2f}]".format(Accuracy))
print("Final Weights =",PClass.get_weights())