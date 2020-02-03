#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron

from perceptron import PerceptronClassifier
from arff import Arff
from graph_tools import *

def basic():
    print("-------------basic---------------")

    x = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    y = np.array([[0],[1],[1],[0]])

    # print(x)
    # print(y)

    pc = PerceptronClassifier(lr=1)
    # print(pc)
    print(pc.fit(x, y).score(np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,0]]), np.array([[0],[0],[1],[1]])))
    print(pc)

def debug():
    print("------------arff-------------------")

    mat = Arff("../data/perceptron/debug/linsep2nonorigin.arff", label_count=1)
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)
    PClass = PerceptronClassifier(
        lr=0.1, shuffle=False, deterministic=10, printIt=False)
    PClass.fit(data, labels)
    Accuracy = PClass.score(data, labels)
    print("Accuray = [{:.2f}]".format(Accuracy))
    print("Final Weights =", PClass.get_weights())
    # print(PClass)

def evaluation():
    print("--------------arf2------------------------------")

    mat = Arff("../data/perceptron/evaluation/data_banknote_authentication.arff", label_count=1)
    np_mat = mat.data
    data = mat[:, :-1]
    labels = mat[:, -1].reshape(-1, 1)

    #### Make Classifier ####
    P2Class = PerceptronClassifier(lr=0.1, shuffle=False, deterministic=10)
    P2Class.fit(data, labels)
    Accuracy = P2Class.score(data, labels)
    print("Accuray = [{:.2f}]".format(Accuracy))
    print("Final Weights =", P2Class.get_weights())
    # print(P2Class)

def separable():
    print("----------------separable-----------------------")

    mat = Arff("./separableIsSquare.arff", label_count=1)
    np_mat = mat.data
    data = mat[:, :-1]
    labels = mat[:, -1].reshape(-1, 1)
    print(data[:, 1])
    print(labels)

    ### Make the Classifier #####
    P3Class = None
    for lr in range(10, 0, -1):
        P3Class = PerceptronClassifier(lr=0.1*lr, shuffle=False)
        P3Class.fit(data, labels, standard_weight_value=None)
        Accuracy = P3Class.score(data, labels)
        print("Learning Rate = ", 0.1*lr)
        print("Accuracy = [{:.2f}]".format(Accuracy))
        print("Epochs = ", P3Class.get_epochs_trained())
    # print(P3Class)


    ## could not get graphing to work in time...
    # graph(data[:, 0], data[:, 1], labels=mat[:, -1])

    w = P3Class.get_weights()
    y = lambda x: (-w[0]/w[1])*x - (w[2]/w[1])

    grapher = Grapher()
    grapher.graph(data[:, 0], data[:, 1], labels=mat[:, -1], title="Separable")
    grapher.add_function(y)

    grapher.show("separable.svg")




def inseparable():
    print("----------------Inseparable-----------------------")

    mat = Arff("./impossible.arff", label_count=1)
    np_mat = mat.data
    data = mat[:, :-1]
    labels = mat[:, -1].reshape(-1, 1)

    ### Make the Classifier #####
    P4Class = None
    for lr in range(10, 0, -1):
        P4Class = PerceptronClassifier(lr=0.1*lr, deterministic=10, shuffle=False)
        P4Class.fit(data, labels, standard_weight_value=None)
        Accuracy = P4Class.score(data, labels)
        print("Learning Rate = ", 0.1*lr)
        print("Accuracy = [{:.2f}]".format(Accuracy))
        print("Epochs = ", P4Class.get_epochs_trained())

    w = P4Class.get_weights()
    y = lambda x: (-w[0]/w[1])*x - (w[2]/w[1])

    grapher = Grapher()
    grapher.graph(data[:, 0], data[:, 1], labels=mat[:, -1], title="Inseparable")
    grapher.add_function(y)

    grapher.show("Inseparable.svg")

    # print(P4Class)

def _shuffle_split(data, percent_test):
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
    for i in indicies:
        training.append(data[i])
    
    testing = []
    for i in poss_indecies:
        testing.append(data[i])


    return np.array(training), np.array(testing)

def voting():
    print("--------------voting---------------------")
    mat = Arff("../data/perceptron/vote.arff", label_count=1)
    np_mat = mat.data

    avg = []

    for iteration in range(5):
        print("xxxxxxxxxxx   " + str(iteration) + "  xxxxxxxx")
        training, testing = _shuffle_split(mat.data, .3)

        data = training[:, :-1]
        labels = training[:, -1].reshape(-1, 1)
        P5Class = PerceptronClassifier(lr=0.1, shuffle=True)
        P5Class.fit(data, labels)

        Accuracy = P5Class.score(data, labels)
        print("Accuracy = [{:.2f}]".format(Accuracy))
        print("Epochs = ", P5Class.get_epochs_trained())    

        tData = testing[:, :-1]
        tLabels = testing[:, -1].reshape(-1, 1)
        tAccuracy = P5Class.score(tData, tLabels)
        print("Test Accuracy = [{:.2f}]".format(tAccuracy))

        weights = P5Class.get_weights()
        print(weights)
        sort_weights = sorted(zip(weights, list(range(len(weights)))), key=lambda x: abs(x[0]), reverse=True)
        print("sorted:\r\n", sort_weights)

        scores = P5Class.getTrace().getColumns("epochScore")
        print('scores', scores)
        avg.append((float(scores[-2][0]) - float(scores[0][0])) / len(scores))
    
    print('avg', avg)
    grapher = Grapher()
    grapher.graph(list(range(len(avg))), avg, labels=[1]*len(avg), points=False, title="Average Scores", xlabel="Iteration", ylabel="score")
    grapher.show("AverageScores.svg")
    

p = Perceptron()
p.f

basic()
debug()
evaluation()
separable()
inseparable()
voting()