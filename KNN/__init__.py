import numpy as np
from KNN import KNNClassifier #as my_knn

from sklearn.preprocessing import Normalizer

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
    print("Accuracy = [{:.2f}]".format(score))

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

    transformer = Normalizer()
    dataN = transformer.transform(data)
    data2N = transformer.transform(data2)    

    classifier = KNNClassifier(weight_type="no_weight")

    classifier.fit(data, labels)
    predicted = np.array(classifier.predict(data2)[0])
    np.savetxt('magic-telescope.csv', predicted, delimiter=',')
    score = classifier.score(data2, labels2, predict_results=predicted)
    print("Accuracy=[{:.2f}]".format(score))
    classifier.fit(dataN, labels)
    predicted = np.array(classifier.predict(data2N)[0])
    np.savetxt('magic-telescopeN.csv', predicted, delimiter=',')
    score = classifier.score(data2N, labels2, predict_results=predicted)
    print("Accuracy N=[{:.2f}]".format(score))

    accuracies = []
    for k in range(1,17,2):
        classifier = KNNClassifier(weight_type="no_weight", k=k)
        accuracies.append(classifier.fit(dataN, labels).score(data2N, labels2))
    np.savetxt('magic-telescopeAccuracies.csv', np.array(accuracies), delimiter=',')


def housingNW():
    print('-----------housing no weight-----------')
    mat = Arff('./data/housing_train.arff', label_count=1)
    mat2 = Arff('./data/housing_test.arff', label_count=1)

    data = mat.data[:,0:-1]
    labels = mat.data[:,-1].reshape(-1, 1)

    data2 = mat2.data[:,0:-1]
    labels2 = mat2.data[:,-1].reshape(-1, 1)

    transformer = Normalizer()
    dataN = transformer.transform(data)
    data2N = transformer.transform(data2)

    classifier = KNNClassifier(regression=True, weight_type="no_weight")

    mse = classifier.fit(dataN, labels).score(data2N, labels2, style="mse")
    print("MSE:", mse)

    mse_scores = []
    for k in range(1,17,2):
        classifier = KNNClassifier(weight_type="no_weight", regression=True, k=k)
        mse_scores.append(classifier.fit(dataN, labels).score(data2N, labels2, style='mse'))
    np.savetxt('housing_mseNW.csv', np.array(mse_scores), delimiter=',')

def housingWW():
    print('-----------housing weighted-----------')
    mat = Arff('./data/housing_train.arff', label_count=1)
    mat2 = Arff('./data/housing_test.arff', label_count=1)

    data = mat.data[:,0:-1]
    labels = mat.data[:,-1].reshape(-1, 1)

    data2 = mat2.data[:,0:-1]
    labels2 = mat2.data[:,-1].reshape(-1, 1)

    transformer = Normalizer()
    dataN = transformer.transform(data)
    data2N = transformer.transform(data2)

    classifier = KNNClassifier(regression=True, weight_type="inverse_distance", labeltype=mat.attr_types)

    mse = classifier.fit(dataN, labels).score(data2N, labels2, style="mse")
    print("MSE:", mse)

    mse_scores = []
    for k in range(1,17,2):
        classifier = KNNClassifier(weight_type="inverse_distance", regression=True, k=k)
        mse_scores.append(classifier.fit(dataN, labels).score(data2N, labels2, style='mse'))
    np.savetxt('housing_mseWW.csv', np.array(mse_scores), delimiter=',')




hwork()
seismic()
evaluation()
magic_telescope()