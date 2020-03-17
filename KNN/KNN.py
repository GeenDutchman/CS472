import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class KNNClassifier(BaseEstimator,ClassifierMixin):


    def __init__(self,labeltype=[],weight_type='inverse_distance'): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal.
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.columntype = columntype
        self.weight_type = weight_type


    def _distance(self, pointOne, pointTwo, diff_exponet=1, rooter=1):
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
        return self
    def predict(self,data):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        pass

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

				return 0


