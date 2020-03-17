import numpy as np
from KNN import KNNClassifier as my_knn

def hwork():
    print('------------hwork-------------')

    data = np.array([[1,5],[0,8],[9,9],[10,10]])
    labels = np.array([['a'],['b'],['b'],['a']])

    classifier = my_knn()

    classifier.fit(data, labels)
    print(classifier(np.array([[2,6]])))


hwork()
