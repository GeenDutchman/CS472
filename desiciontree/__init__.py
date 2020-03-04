import numpy as np
from sklearn.model_selection import train_test_split


from desiciontree import DTClassifier
from arff import Arff

def basic():
    print("---------------basic--------------")
    a = np.array([['Y', 'Thin', 'N', 'Great'],
                  ['N', 'Deep', 'N', 'Bad'],
                  ['N', 'Stuffed', 'Y', 'Good'],
                  ['Y', 'Stuffed', 'Y', 'Great'],
                  #['Y', 'Deep', 'N', 'Good'],
                  ['Y', 'Deep', 'Y', 'Great'],
                  ['N', 'Thin', 'Y', 'Good'],
                  ['Y', 'Deep', 'N', 'Good'],
                  ['N', 'Thin', 'N', 'Bad']])

    data = a[:, 0:-1]
    labels = a[:, -1].reshape(-1, 1)

    # print(data, labels)
    # for index in range(np.shape(a)[1]):
    #     values, counts = np.unique(a[:,index], return_counts=True)
    #     print(values, counts)

    classifier = DTClassifier(0)
    classifier.fit(data, labels)
    lame_test = [['Y', 'Deep', 'N']]
    results = classifier.predict(lame_test)
    print(classifier.tree)
    print(results)
    print(classifier.score(lame_test, [['Good']]))

def lenses():
    print("----------------lenses------------------")

    mat = Arff("./lenses.arff", label_count=1)
    # data = mat.data[:, 0:-1]
    # labels = mat.data[:, -1].reshape(-1, 1)

    data, tData, labels, tLabels = train_test_split(mat.data[:, :-1], mat.data[:, -1].reshape(-1, 1), test_size=.25)

    dtree = DTClassifier(0)
    dtree.fit(data, labels)
    print(dtree.tree)

    results = dtree.predict(tData)
    # for r, t in zip(results, tLabels):
    #     print(r, t)

    score = dtree.score(tData, tLabels)
    print("Accuracy=[{:.2f}]".format(score))

def all_lenses():
    print("---------all-lenses----------")

    lens_data = Arff("./lenses.arff", label_count=1)
    all_lens_data = Arff("./all_lenses.arff", label_count=1)

    lens_split = train_test_split(lens_data.data[:, :-1], lens_data.data[:, -1].reshape(-1, 1), test_size=0.0)#[0, 2]
    all_lens_split = train_test_split(all_lens_data.data[:, :-1], all_lens_data.data[:, -1].reshape(-1, 1), test_size=0)#[1, 3]

    # print(lens_split[0], lens_split[2])
    # print(all_lens_split[0], all_lens_split[2])

    dtree = DTClassifier()
    dtree.fit(lens_split[0], lens_split[2])
    score = dtree.score(all_lens_split[0], all_lens_split[2])
    print("Accuracy=[{:.2f}]".format(score))

def nan_lenses():
    print("----------------nan_lenses------------------")

    mat = Arff("./nan_lenses.arff", label_count=1)
    # data = mat.data[:, 0:-1]
    # labels = mat.data[:, -1].reshape(-1, 1)

    data, tData, labels, tLabels = train_test_split(mat.data[:, :-1], mat.data[:, -1].reshape(-1, 1), test_size=.25)

    dtree = DTClassifier(0)
    dtree.fit(data, labels)
    print(dtree.tree)

    results = dtree.predict(tData)
    # for r, t in zip(results, tLabels):
    #     print(r, t)

    score = dtree.score(tData, tLabels)
    print("Accuracy=[{:.2f}]".format(score))

def evaluation():
    print("----------------evaluation---------------")

    zoo_data = Arff("./zoo.arff", label_count=1)
    all_zoo_data = Arff("./all_zoo.arff", label_count=1)

    zoo_split = train_test_split(zoo_data.data[:, :-1], zoo_data.data[:, -1].reshape(-1, 1), test_size=0.0)#[0, 2]
    all_zoo_split = train_test_split(all_zoo_data.data[:, :-1], all_zoo_data.data[:, -1].reshape(-1, 1), test_size=0)#[1, 3]

    # print(zoo_split[0], zoo_split[2])
    # print(all_zoo_split[0], all_zoo_split[2])

    dtree = DTClassifier()
    dtree.fit(zoo_split[0], zoo_split[2])
    predicted = dtree.predict(all_zoo_split[0])
    np.savetxt('predicted_zoo.csv', predicted, delimiter=',')
    score = dtree.score(all_zoo_split[0], all_zoo_split[2])
    print("Accuracy=[{:.2f}]".format(score))

def cars():
    print("----------------cars------------------")

    mat = Arff("./cars.arff", label_count=1)
    # data = mat.data[:, 0:-1]
    # labels = mat.data[:, -1].reshape(-1, 1)

    data, tData, labels, tLabels = train_test_split(mat.data[:, :-1], mat.data[:, -1].reshape(-1, 1), test_size=.25)

    dtree = DTClassifier()
    dtree.fit(data, labels)

    results = dtree.predict(tData)
    np.savetxt('cars.csv', results, delimiter=',')

    score = dtree.score(tData, tLabels)
    print("Accuracy=[{:.2f}]".format(score))

def voting():
    print("----------------voting------------------")

    mat = Arff("./voting.arff", label_count=1)
    # data = mat.data[:, 0:-1]
    # labels = mat.data[:, -1].reshape(-1, 1)

    data, tData, labels, tLabels = train_test_split(mat.data[:, :-1], mat.data[:, -1].reshape(-1, 1), test_size=.25)
    # print(data)

    dtree = DTClassifier()
    dtree.fit(data, labels)

    results = dtree.predict(tData)
    np.savetxt('voting.csv', results, delimiter=',')

    score = dtree.score(tData, tLabels)
    print("Accuracy=[{:.2f}]".format(score))


basic()
lenses()
nan_lenses()
all_lenses()
evaluation()
cars()
voting()
