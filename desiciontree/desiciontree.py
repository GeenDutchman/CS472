import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from tree import Tree

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score

class DTClassifier(BaseEstimator,ClassifierMixin):


        

    def __init__(self,counts=None):
        """ Initialize class with chosen hyperparameters.

        Args:
            counts = how many types for each attribute
        Example:
            DT  = DTClassifier()
        """
        self.counts = counts
        self.tree = Tree()

    def _most_common_class(self, label_results):
        highest_index = np.argmax(label_results[1])
        return label_results[0][highest_index]

    def _fit(self, X, y, branch, indexes):
        best_branch = (0, None)
        out_of, data_results, label_results = self._count_unique(X, y)
        entropy = self._entropy(out_of, label_results)

        for i in indexes:
            partitions = self._partition(X, y, i, data_results=data_results)
            sum = 0
            for part in partitions:
                print(part, best_branch)
                y_shape = np.shape(part[1])[0]
                fraction = y_shape / out_of
                sum = sum + self._entropy(y_shape, part[1]) * fraction
                gain = entropy - sum
                a_branch = Tree.Branch(self._most_common_class(part[1]), i, data_results[0])
                if gain > best_branch[0]:
                    best_branch = (gain, a_branch)
        print(best_branch)

    def fit(self, X, y):
        """ Fit the data; Make the Desicion tree

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        indexes = {x for x in range(np.shape(X)[1])}
        self._fit(X, y, None, indexes)


        # use LabelBinarizer?

        return self

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        pass


    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-li    def _shuffle_data(self, X, y):
        """
        return 0

    def _count_unique(self, X, y):
        data_results = {}
        for index in range(np.shape(X)[1]):
            values, counts = np.unique(X[:,index], return_counts=True)
            data_results[index] = (values, counts)

        label_results = {}
        for index in range(np.shape(y)[1]):
            values, counts = np.unique(y[:,index], return_counts=True)
            label_results[index] = (values, counts)

        return np.shape(y)[0], data_results, label_results

    def _partition(self, X, y, attribute_index, data_results=None):
        if data_results is None:
            data_results = self._count_unique(X, y)[1]
        partitions = []
        for label in data_results[attribute_index][0]:
            x_part = []
            y_part = []
            for data_point_index in range(np.shape(X)[0]):
                if label == X[data_point_index][attribute_index]:
                    x_part.append(X[data_point_index])
                    y_part.append(y[data_point_index])
            partitions.append((np.array(x_part), np.array(y_part)))

        return partitions

    def _entropy(self, out_of, values_and_counts):
        sum = 0
        for column in values_and_counts:
            for count in values_and_counts[column][1]:
                fraction = count / out_of
                sum = sum - fraction * np.log2(fraction)
        return sum

