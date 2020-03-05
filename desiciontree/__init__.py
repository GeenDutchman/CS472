import numpy as np
from sklearn.model_selection import train_test_split, KFold


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

    # results = dtree.predict(tData)
    # for r, t in zip(results, tLabels):
    #     print(r, t)

    score = dtree.score(tData, tLabels)
    print("Accuracy=[{:.2f}]".format(score))

def all_lenses():
    print("---------all-lenses----------")

    lens_data = Arff("./lenses.arff", label_count=1)
    all_lens_data = Arff("./all_lenses.arff", label_count=1)

    lens_train = lens_data.data[:, :-1]
    lens_label_train = lens_data.data[:, -1].reshape(-1, 1)
    lens_test = all_lens_data.data[:, :-1]
    lens_label_test = all_lens_data.data[:, -1].reshape(-1, 1)

    dtree = DTClassifier()
    dtree.fit(lens_train, lens_label_train)
    score = dtree.score(lens_test, lens_label_test)
    print("Train Accuracy=[{:.2f}]".format(dtree.score(lens_train, lens_label_train)))
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

    # results = dtree.predict(tData)
    # for r, t in zip(results, tLabels):
    #     print(r, t)

    score = dtree.score(tData, tLabels)
    print("Accuracy=[{:.2f}]".format(score))

def evaluation():
    print("----------------evaluation---------------")

    zoo_data = Arff("./zoo.arff", label_count=1)
    all_zoo_data = Arff("./all_zoo.arff", label_count=1)

    zoo_train = zoo_data.data[:, :-1]
    zoo_label_train = zoo_data.data[:, -1].reshape(-1, 1)
    zoo_test = all_zoo_data.data[:, :-1]
    zoo_label_test = all_zoo_data.data[:, -1].reshape(-1, 1)

    dtree = DTClassifier()
    dtree.fit(zoo_train, zoo_label_train)
    print("Train Accuracy=[{:.2f}]".format(dtree.score(zoo_train, zoo_label_train)))

    predicted = dtree.predict(zoo_test)
    np.savetxt('predicted_zoo.csv', predicted, delimiter=',')
    score = dtree.score(zoo_test, zoo_label_test)
    print("Accuracy=[{:.2f}]".format(score))

def cars():
    print("----------------cars------------------")

    mat = Arff("./cars.arff", label_count=1)
    # data = mat.data[:, 0:-1]
    # labels = mat.data[:, -1]#.reshape(-1, 1)
    splits = 10
    kfolder = KFold(n_splits=splits)

    scores = []

    data, tData, labels, tLabels = train_test_split(mat.data[:, :-1], mat.data[:, -1].reshape(-1, 1), test_size=.25)
    best_tree = (0, None)
    for train, validate in kfolder.split(data, labels):
        # print(train, validate)
        dtree = DTClassifier()
        dtree.fit(data[train], labels[train])

        scores.append(dtree.score(data[validate], labels[validate]))
        if scores[-1] > best_tree[0]:
            best_tree = (scores[-1], dtree)

    average = np.sum(scores) / splits
    scores.append(average)
    header_text = ''
    for x in range(splits):
        header_text = header_text + str(x) + ' '
    
    np.savetxt("cars.csv", scores, header=header_text + 'average', delimiter=',')
    print(scores)
    print('Best tree accuracy: {:.2f}'.format(best_tree[1].score(tData, tLabels)))

def voting():
    print("----------------voting------------------")

    mat = Arff("./voting.arff", label_count=1)
     # data = mat.data[:, 0:-1]
    # labels = mat.data[:, -1]#.reshape(-1, 1)
    splits = 10
    kfolder = KFold(n_splits=splits)

    scores = []

    data, tData, labels, tLabels = train_test_split(mat.data[:, :-1], mat.data[:, -1].reshape(-1, 1), test_size=.25)
    best_tree = (0, None)
    for train, validate in kfolder.split(data, labels):
        # print(train, validate)
        dtree = DTClassifier()
        dtree.fit(data[train], labels[train])

        scores.append(dtree.score(data[validate], labels[validate]))
        if scores[-1] > best_tree[0]:
            best_tree = (scores[-1], dtree)

    average = np.sum(scores) / splits
    scores.append(average)
    header_text = ''
    for x in range(splits):
        header_text = header_text + str(x) + ' '
    
    np.savetxt("voting.csv", scores, header=header_text + 'average', delimiter=',')
    print(scores)
    print('Best tree accuracy: {:.2f}'.format(best_tree[1].score(tData, tLabels)))


basic()
lenses()
nan_lenses()
all_lenses()
evaluation()
cars()
voting()
