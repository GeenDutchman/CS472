#!/bin/python3

import numpy as np

from perceptron import PerceptronClassifier
from arff import Arff

def basic():
    print("-------------basic---------------")

    x = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    y = np.array([[0],[1],[1],[0]])

    # print(x)
    # print(y)

    pc = PerceptronClassifier(lr=1)
    # print(pc)
    print(pc.fit(x, y).score(np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,0]]), np.array([[0],[0],[1],[1]])))
    # print(pc)

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

def seperable():
    print("----------------seperable-----------------------")

    mat = Arff("./seperableIsSquare.arff", label_count=1)
    np_mat = mat.data
    data = mat[:, :-1]
    labels = mat[:, -1].reshape(-1, 1)

    ### Make the Classifier #####
    P3Class = None
    for lr in range(10, 0, -1):
        P3Class = PerceptronClassifier(lr=0.1*lr, shuffle=False)
        P3Class.fit(data, labels)
        Accuracy = P3Class.score(data, labels)
        print("Learning Rate = ", 0.1*lr)
        print("Accuracy = [{:.2f}]".format(Accuracy))
        print("Epochs = ", P3Class.get_epochs_trained())
    # print(P3Class)

def inseperable():
    print("----------------Inseperable-----------------------")

    mat = Arff("./impossible.arff", label_count=1)
    np_mat = mat.data
    data = mat[:, :-1]
    labels = mat[:, -1].reshape(-1, 1)

    ### Make the Classifier #####
    P4Class = None
    for lr in range(10, 0, -1):
        P4Class = PerceptronClassifier(lr=0.1*lr, deterministic=10, shuffle=False)
        P4Class.fit(data, labels)
        Accuracy = P4Class.score(data, labels)
        print("Learning Rate = ", 0.1*lr)
        print("Accuracy = [{:.2f}]".format(Accuracy))
        print("Epochs = ", P4Class.get_epochs_trained())

    # print(P4Class)
basic()
debug()
evaluation()
seperable()
inseperable()