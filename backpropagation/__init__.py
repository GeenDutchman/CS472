#!/bin/python3

import numpy as np
from mlp import MLPClassifier
from arff import Arff

from sklearn import preprocessing
from sklearn.model_selection import train_test_split



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
    print("------------arff-------------------")

    mat = Arff("../data/perceptron/debug/linsep2nonorigin.arff", label_count=1)
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)
    # print("data\n", data)
    MLPClass = MLPClassifier([2*np.shape(data)[1]], lr=0.1, shuffle=False, deterministic=10)
    MLPClass.fit(data, labels, momentum=0.5, percent_verify=0, standard_weight=0)
    Accuracy = MLPClass.score(data, labels)
    # print(MLPClass)
    retrieved_weights = MLPClass.get_weights()

    for layer in range(len(retrieved_weights)):
        np.savetxt("linsep_weights_eval_" + str(layer) + ".csv", retrieved_weights[layer], delimiter=',')


def evaluate():
    print("------------eval-------------------")

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
    training = np.array(training)
    training_target = np.array(training_target)
    
    testing = []
    testing_target = []
    for i in poss_indecies:
        testing.append(data[i])
        testing_target.append(targets[i])
    testing = np.array(testing)
    testing_target = np.array(testing_target)

    non_duplicate = 0
    for point in reversed(testing):
        if point not in training:
            non_duplicate = non_duplicate + 1
    non_duplicate1 = 0
    for point in reversed(training):
        if point not in testing:
            non_duplicate1 = non_duplicate1 + 1

    assert(non_duplicate != 0)
    assert(non_duplicate1 != 0)

    # print("training", len(training), "t_train", len(training_target), "testing", len(testing), "t_target", len(testing_target))
    return training, training_target, testing, testing_target

def iris():
    print("-------------iris----------------")
    mat = Arff("../data/perceptron/iris.arff", label_count=3)

    y = mat.data[:,-1]
    # print(y)

    lb = preprocessing.LabelBinarizer()
    lb.fit(y)
    y = lb.transform(y)

    # split it
    # data, labels, tData, tLabels = _shuffle_split(mat.data[:, :-1], y, .25)
    data, tData, labels, tLabels = train_test_split(mat.data[:, :-1], y, test_size=.25)

    MLPClass = MLPClassifier([2*np.shape(data)[1]], lr=0.1, shuffle=True, one_hot=True)
    MLPClass.fit(data, labels, momentum=0.5, percent_verify=.25)

    np.savetxt("Iris_eval.csv", MLPClass.stupidData[1:], header=reduce(MLPClass.stupidData[0]), delimiter=',')

    accuracy = MLPClass.score(tData, tLabels)
    print("Test Accuracy = [{:.2f}]".format(accuracy))

def reduce(array, delim=','):
    out = ''
    for x in array:
        out = out + str(x) + delim
    return out

def vowel():
    print("-------------vowel----------------")
    mat = Arff("../data/vowel.arff", label_count=1)
    y = mat.data[:,-1]
    lb = preprocessing.LabelBinarizer()
    lb.fit(y)
    y = lb.transform(y)

    # split it
    data, tData, labels, tLabels = train_test_split(mat.data[:, :-1], y, test_size=.25)

    master_window = 5
    window = master_window
    bssf = [np.inf, 0]
    tolerance = 1e-4
    findings = []
    findings.append(["LR", "Epochs", "TestAccuracy", "MSE train", "MSE validate", "MSE test", "Best LR"])
    for lr in range(-1, 5):
        if window <= 0:
            break
        for step in [1, 5]:
            entry = []
            print((lr, step), end=",")
            learn_rate = (0.1**lr)*step
            MLPClass = MLPClassifier([2*np.shape(data)[1]], lr=learn_rate, shuffle=True, one_hot=True)
            MLPClass.fit(data, labels, momentum=0.5, percent_verify=.25)

            accuracy = MLPClass.score(tData, tLabels)
            entry.append(learn_rate)
            entry.append(MLPClass.getEpochCount())
            entry.append(accuracy)
            entry.append(MLPClass._calc_l2_err(data, labels))
            entry.append(MLPClass.bssf[0])
            entry.append(MLPClass._calc_l2_err(tData, tLabels))
            entry.append(bssf[1])

            findings.append(entry)

            if accuracy < bssf[0] and abs(accuracy - bssf[0]) > tolerance:
                bssf = [accuracy, learn_rate]
                window = master_window
            else:
                window = window - 1
                if window <= 0:
                    break

    print("\n\r", findings)
    np.savetxt("vowel_findings_lr.csv", findings[1:], header=reduce(findings[0]), delimiter=',')

    lr = bssf[1]
    window = master_window
    findings = []
    findings.append(["Num Nodes", "Epochs", "Train Accuracy", 'VS accuracy', 'test accuracy'])
    accuracy = bssf[0]
    doubler = 0
    num_nodes = 0
    bssf = [np.inf, 0]
    while(window > 0):
        num_nodes = num_nodes * doubler
        print("numnodes", num_nodes)

        MLPClass = MLPClassifier([num_nodes], lr=lr, shuffle=True, one_hot=True)
        MLPClass.fit(data, labels, momentum=0.5, percent_verify=.25)

        accuracy = MLPClass.score(tData, tLabels)
        entry = []

        entry.append(num_nodes)
        entry.append(MLPClass.getEpochCount())
        entry.append(MLPClass._calc_l2_err(data, labels))
        entry.append(MLPClass.bssf[0])
        entry.append(MLPClass._calc_l2_err(tData, tLabels))

        findings.append(entry)

        if accuracy < bssf[0] and abs(accuracy - bssf[0]) > tolerance:
            bssf = [accuracy, num_nodes]
            window = master_window
        else:
            window = window - 1

    np.savetxt("vowel_findings_hid_nodes.csv", findings[1:], header=reduce(findings[0]), delimiter=',')

    num_nodes = bssf[1]
    window = master_window
    findings = []
    findings.append(["Momentum", "Epochs", "Train Accuracy", 'VS accuracy', 'test accuracy'])
    momentum = 0
    bssf = [np.inf, momentum]
    while(window > 0):
        momentum = momentum + 0.05
        print("momentum", momentum)

        MLPClass = MLPClassifier([num_nodes], lr=lr, shuffle=True, one_hot=True)
        MLPClass.fit(data, labels, momentum=momentum, percent_verify=.25)

        accuracy = MLPClass.score(tData, tLabels)
        entry = []

        entry.append(momentum)
        entry.append(MLPClass.getEpochCount())
        entry.append(MLPClass._calc_l2_err(data, labels))
        entry.append(MLPClass.bssf[0])
        entry.append(MLPClass._calc_l2_err(tData, tLabels))

        findings.append(entry)

        if accuracy < bssf[0] and abs(accuracy - bssf[0]) > tolerance:
            bssf = [accuracy, momentum]
            window = master_window
        else:
            window = window - 1   

    np.savetxt("vowel_findings_momentum.csv", findings[1:], header=reduce(findings[0]), delimiter=',')



# basic()
# debug()
# evaluate()
# iris()
vowel()
