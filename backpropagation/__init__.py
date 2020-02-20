#!/bin/python3

import numpy as np
from mlp import MLPClassifier
from arff import Arff

from sklearn import preprocessing



def basic():
    print("-------------basic---------------")

    x = np.array([[0,0],[0,1], [1, 1]])
    y = np.array([[1],[0], [0]])
    l1 = np.array([[1, 1], [1, 1], [1, 1]])
    l2 = np.array([[1], [1], [1]])
    w = [l1, l2]
    tx = np.array([[1, 0]]) # second has a zero
    ty = np.array([[1]])

    # print(x)
    # print(y)
    pc = None
    try:
        pc = MLPClassifier([2], 1, shuffle=False, deterministic=10)
        # print(pc)
        print(pc.fit(x, y, w, 1, percent_verify=0))
        print("fake score", pc.score(tx, ty))
        # print(pc.fit(x, y).score(np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,0]]), np.array([[0],[0],[1],[1]])))
        # print(pc)
    except Exception as e:
        print("AN EXCEPTION OCCURRED!!!---------\n\r"*2, pc)
        raise e

def debug():
    # print("------------arff-------------------")

    mat = Arff("../data/perceptron/debug/linsep2nonorigin.arff", label_count=1)
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)
    # print("data\n", data)
    MLPClass = MLPClassifier([2*np.shape(data)[1]], lr=0.1, shuffle=False, deterministic=10)
    MLPClass.fit(data, labels, momentum=0.5, percent_verify=0, standard_weight=0)
    Accuracy = MLPClass.score(data, labels)
    # print(MLPClass)
    print("Final Weights =", MLPClass.get_weights())
    # print(MLPClass)

def eval():
    # print("------------eval-------------------")

    mat = Arff("../data/perceptron/evaluation/data_banknote_authentication.arff", label_count=1)
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)
    # print("data\n", data)
    MLPClass = MLPClassifier([2*np.shape(data)[1]], lr=0.1, shuffle=False, deterministic=10)
    MLPClass.fit(data, labels, momentum=0.5, percent_verify=0, standard_weight=0)
    Accuracy = MLPClass.score(data, labels)
    # print(MLPClass)
    # print("Final Weights =", MLPClass.get_weights())
    # print(MLPClass)
    print(MLPClass.csv_print())

def _shuffle_split(data, targets, percent_test):
    poss_indecies = list(range(len(data)))
    indicies = []
    print(len(poss_indecies))
    print(int(percent_test * len(poss_indecies)))
    testSize = int(percent_test * len(poss_indecies))
    while len(poss_indecies) > testSize:
        index = np.random.randint(0, len(poss_indecies))
        result = poss_indecies.pop(index)
        # indicies.append(result)
        # print(index, result)
        indicies.append(result)

    training = []
    training_target = []
    for i in indicies:
        training.append(data[i])
        training_target.append(targets[i])
    
    testing = []
    testing_target = []
    for i in poss_indecies:
        testing.append(data[i])
        testing_target.append(targets[i])


    return np.array(training), np.array(training_target), np.array(testing), np.array(testing_target)

def iris():
    # print("-------------iris----------------")
    mat = Arff("../data/perceptron/iris.arff", label_count=3)

    y = mat.data[:,-1]
    print(y)

    lb = preprocessing.LabelBinarizer()
    lb.fit(y)
    y = lb.transform(y)

    # split it
    data, labels, tData, tLabels = _shuffle_split(mat.data, y, .25)


    MLPClass = MLPClassifier([2*np.shape(data)[1]], lr=0.1, shuffle=True)
    MLPClass.fit(data, labels, momentum=0.5, percent_verify=.25)

    accuracy = MLPClass.score(tData, tLabels)
    print("accuracy: ", accuracy)


# basic()
# debug()
# eval()
iris()
