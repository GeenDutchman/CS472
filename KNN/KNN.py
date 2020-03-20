import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from itertools import zip_longest

class KNNClassifier(BaseEstimator,ClassifierMixin):


    def __init__(self,labeltype=[],weight_type='inverse_distance',k=3,regression=False): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal.
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.regression = regression
        self.columntype = labeltype
        self.weight_type = weight_type
        self._default_diff = 1
        self._default_same = 0
        self.k = k


    def _distance(self, pointOne, pointTwo, diff_exponet=2, rooter=0.5):
        total = 0
        for x, y, col_type in zip_longest(pointOne, pointTwo, self.columntype, fillvalue="real"):
            # if type(x) != type(y):
            #     total += self._default_diff ** diff_exponet
            # else:
            try:
                total += abs(x - y) ** diff_exponet
            except TypeError:
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
        if not self.regression:
            for neighbor in neighbors:
                this_neighbor = 1 if self.weight_type == "no_weight" else inverse_distance(neighbor[0])
                if neighbor[1] in results:
                    results[neighbor[1]] += this_neighbor
                else:
                    results[neighbor[1]] = this_neighbor

            return max(results, key=lambda k: results[k])
        else:
            inverses = []
            numerator = np.zeros(np.shape(neighbors[0][2]))
            for neighbor in neighbors:
                if self.weight_type == 'inverse_distance':
                    inverses.append(inverse_distance(neighbor[0]))
                else:
                    inverses.append(1)
                numerator = numerator + (inverses[-1] * neighbor[2])

            return numerator / np.sum(inverses)


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
                neighbors.append((self._distance(self.data[stored_index], data[predict_index]), self.data_labels[stored_index][0], data[predict_index]))
            neighbors = sorted(neighbors, key=lambda x: x[0])[:self.k]
            results.append(self._analyze_neighbors(neighbors))   

## This is code that I was trying to get optimized...I didn't allow for enough time
'''
        distances = [np.linalg.norm(self.data - x, axis=1) for x in data]
        if self.weight_type == "inverse_distance":
            inverse = lambda x : x**-2
            vectorized = np.vectorize(inverse)
            distances = [vectorized(x) for x in distances]
        argDist = [np.argpartition(x,len(x) - (self.k + 1))[:self.k] for x in distances]
        
        if not self.regression:
            results = []
            for pointIndex in range(len(argDist)):
                neighbor_label = {}
                point = argDist[pointIndex]
                for neighborIndex in point:
                    this_neighbor = 1 if self.weight_type == "inverse_distance" else distances[pointIndex][neighborIndex]
                    label = self.data_labels[neighborIndex][0]
                    if label in neighbor_label:
                        neighbor_label[label] = this_neighbor
                    else:
                        neighbor_label[label] += this_neighbor
                results.append(max(results, key=lambda k: results[k]))
        else:
            numerator = np.zeros(np.shape(distances[0][0]))
            for pointIndex in range(len(argDist)):
                point = argDist[pointIndex]
                for neighborIndex in point:
                    dist = distances[pointIndex][neighborIndex]
'''        

        
        return results, np.shape(results)

    #Returns the Mean score given input data and labels
    def score(self, X, y, predict_results=None, style="accuracy"):
            """ Return accuracy of model on a given dataset. Must implement own score function.
            Args:
                    X (array-like): A 2D numpy array with data, excluding targets
                    y (array-like): A 2D numpy array with targets
            Returns:
                    score : float
                            Mean accuracy of self.predict(X) wrt. y.
            """
            results = predict_results
            if results is None:
                results = np.reshape(self.predict(X)[0], np.shape(y))
            if style=='accuracy':
                correct = 0
                for scored, expected in zip(results, y):
                    if scored == expected:
                        correct += 1
                return 0 if len(results) == 0 else (correct / len(results)) * 100.0
            if style=='mse':
                summer = 0
                count = 0
                for scored, expected in zip(results, y):
                    summer = summer + ((scored - expected) ** 2)
                    count = count + 1
                return summer / count


