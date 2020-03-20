import numpy as np
from KNN import KNNClassifier #as my_knn

from arff import Arff

def hwork():
    print('------------hwork-------------')

    data = np.array([[1,5],[0,8],[9,9],[10,10]])
    labels = np.array([['a'],['b'],['b'],['a']])

    classifier = KNNClassifier()

    classifier.fit(data, labels)
    print(classifier.predict(np.array([[2,6]])))

def seismic():
    print('--------------seismic------------')
    mat = Arff('./data/seismic-bumps_train.arff', label_count=1)
    mat2 = Arff('./data/seismic-bumps_test.arff', label_count=1)

    data = mat.data[:,0:-1]
    labels = mat.data[:,-1].reshape(-1, 1)

    data2 = mat2.data[:,0:-1]
    labels2 = mat2.data[:,-1].reshape(-1, 1)

    classifier = KNNClassifier()

    classifier.fit(data, labels)
    score = classifier.score(data2, labels2)
    print("Accuracy=[{:.2f}]".format(score))

def evaluation():
    print("--------------evaluation-----------------")
    mat = Arff('./data/diabetes.arff', label_count=1)
    mat2 = Arff('./data/diabetes_test.arff', label_count=1)

    data = mat.data[:,0:-1]
    labels = mat.data[:,-1].reshape(-1, 1)

    data2 = mat2.data[:,0:-1]
    labels2 = mat2.data[:,-1].reshape(-1, 1)

    classifier = KNNClassifier()

    classifier.fit(data, labels)
    predicted = np.array(classifier.predict(data2)[0])
    np.savetxt('evaluation.csv', predicted, delimiter=',')
    score = classifier.score(data2, labels2, predict_results=predicted)
    print("Accuracy=[{:.2f}]".format(score))


def magic_telescope():
    print("---------------magic-telescope------------")
    mat = Arff('./data/magic-telescope_train.arff', label_count=1)
    mat2 = Arff('./data/magic-telescope_test.arff', label_count=1)

    data = mat.data[:,0:-1]
    labels = mat.data[:,-1].reshape(-1, 1)

    data2 = mat2.data[:,0:-1]
    labels2 = mat2.data[:,-1].reshape(-1, 1)

    

    classifier = KNNClassifier()

    classifier.fit(data, labels)
    predicted = np.array(classifier.predict(data2)[0])
    np.savetxt('magic-telescope.csv', predicted, delimiter=',')
    score = classifier.score(data2, labels2, predict_results=predicted)
    print("Accuracy=[{:.2f}]".format(score))


hwork()
seismic()
evaluation()
magic_telescope()