import numpy as np

from desiciontree import DTClassifier

def basic():
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
    results = classifier.predict([['Y', 'Deep', 'N', 'Good']])
    print(results)

basic()
