import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class KNNClassifier(BaseEstimator,ClassifierMixin):


    def __init__(self,labeltype=[],weight_type='inverse_distance',k=3): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal.
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.columntype = labeltype
        self.weight_type = weight_type
        self._default_diff = 1
        self._default_same = 0
        self.k = k


    def _distance(self, pointOne, pointTwo, diff_exponet=2, rooter=0.5):
        total = 0
        for x, y in zip(pointOne, pointTwo):
            if type(x) != type(y):
                total += self._default_diff ** diff_exponet
            elif isinstance(x, (float, int)):
                total += abs(x - y) ** diff_exponet
            else:
                total += self._default_diff  ** diff_exponet if x != y else self._default_same ** diff_exponet 

        return total ** float(rooter)


    def fit(self,data,labels):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.data = data
        self.data_labels = labels
        return self

    def _analyze_neighbors(self, neighbors):
        inverse_distance = lambda dist: dist ** -2
        results = {}
        for neighbor in neighbors:
            if neighbor[1] in results:
                results[neighbor[1]] += 1 if self.weight_type == "no_weight" else inverse_distance(neighbors[0])
            else:
                results[neighbor[1]] = 1 if self.weight_type == "no_weight" else inverse_distance(neighbor[0])

        return max(results, key=lambda k: results[k])


    def predict(self,data):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        results = []
        predict_instances = np.shape(data)[0]
        stored_instances = np.shape(self.data)[0]
        for predict_index in range(predict_instances):
            neighbors = [] # dist, label
            for stored_index in range(stored_instances):
                neighbors.append((self._distance(self.data[stored_index], data[predict_index]), self.data_labels[stored_index]))
            neighbors = sorted(neighbors)[:self.k]
            results.append(self._analyze_neighbors(neighbors))   
        
        return results, np.shape(results)

    #Returns the Mean score given input data and labels
    def score(self, X, y):
            """ Return accuracy of model on a given dataset. Must implement own score function.
            Args:
                    X (array-like): A 2D numpy array with data, excluding targets
                    y (array-like): A 2D numpy array with targets
            Returns:
                    score : float
                            Mean accuracy of self.predict(X) wrt. y.
            """
            results = self.predict(X)
            correct = 0
            for scored, expected in zip(results, y):
                if scored == expected:
                    correct += 1
            return 0 if len(results) == 0 else correct / len(results)


