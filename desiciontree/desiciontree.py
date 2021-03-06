import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from tree import Tree

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score

class DTClassifier(BaseEstimator,ClassifierMixin):        

    def __init__(self,counts=None,features=None):
        """ Initialize class with chosen hyperparameters.

        Args:
            counts = how many types for each attribute
        Example:
            DT  = DTClassifier()
        """
        self.features = features
        self.counts = counts
        self.tree = Tree()
        self.nan_replace = np.nan

    def _most_common_class(self, label_results):
        if len(label_results[1]) == 0:
            return None
        highest_index = np.argmax(label_results[1])
        return label_results[0][highest_index]

    def _stopper(self, label_size, indexes, data_size):
        if label_size <= 1:
            return True
        if len(indexes) <= 1:
            return True
        if data_size <= 1:
            return True
        return False

    def _fit(self, X, y, indexes):
        out_of, data_results = self._count_unique(X)
        label_size, label_results = self._count_unique(y)
        if self._stopper(len(label_results[0][0]), indexes, out_of):
            index = -1 if len(indexes) < 1 else indexes.pop()
            return Tree().makeAddBranch(None, None, self._most_common_class(label_results[0]), index, [], friendly_split=self.features[index] if (self.features is not None and len(self.features) > 0 and len(self.features) > abs(index)) else None)
        entropy = self._entropy(label_size, label_results)

        best_branch = (0, None, 0, []) # gain, tree, index, partitions
        for i in indexes:
            # partition based on index
            partitions = self._partition(X, y, i, data_results=data_results)
            partitions_entropy = self._partition_entropy(partitions, out_of)
            gain = entropy - partitions_entropy
            #find partition with best gain
            if gain >= best_branch[0]:
                friendly_split = None
                if self.features is not None:
                    friendly_split = self.features[i]
                child_tree = Tree().makeAddBranch(None, None, self._most_common_class(label_results[0]), i, data_results[i][0], friendly_split=friendly_split)
                best_branch = (gain, child_tree, i, partitions)

        # don't look at the index we just did anymore
        to_send_copy = indexes.copy()
        to_send_copy.remove(best_branch[2])
        for part in best_branch[3]: #per attribute
            grandchild_tree = self._fit(part[0], part[1], to_send_copy)
            parent_partition = part[0][0][best_branch[2]]
            best_branch[1].addChildTree(grandchild_tree, parent_partition)
        return best_branch[1]

    def fit(self, X, y):
        """ Fit the data; Make the Desicion tree

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        indexes = {x for x in range(np.shape(X)[1])}
        self.tree = self._fit(X, y, indexes)

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
        results = []
        for x in X:
            results.append(self.tree.traverse(x))
        return np.reshape(results, (len(results), -1))


    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array of the targets
        """
        predicted = self.predict(X)
        count = 0
        for predict, expected in zip(predicted, y):
            if predict == expected:
                count = count + 1
            
        return count / np.shape(y)[0]

    def _count_unique(self, data):
        data_results = {}
        if np.shape(data)[0] == 0:
            data_results[0] = ([], [])
            return 0, data_results
        for index in range(np.shape(data)[1]):
            #find all unique non-nan values
            column = data[:,index]
            # https://stackoverflow.com/a/37148508 use nan != nan
            values, counts = np.unique([x for x in column if x==x], return_counts=True)
            nan_sum = np.size([x for x in column if x!=x])
            if nan_sum > 0:
                values = np.append(values, self.nan_replace)
                counts = np.append(counts, nan_sum)
            data_results[index] = (values, counts)

        return np.shape(data)[0], data_results

    def _partition(self, X, y, attribute_index, data_results=None):
        if data_results is None:
            data_results = self._count_unique(X)[1]
        partitions = []
        for label in data_results[attribute_index][0]:
            x_part = []
            y_part = []
            for data_point_index in range(np.shape(X)[0]):
                particular_point_label = X[data_point_index][attribute_index]
                if label == particular_point_label:
                    x_part.append(X[data_point_index])
                    y_part.append(y[data_point_index])
                elif (label == self.nan_replace or label != label) and np.isnan(particular_point_label):
                    x_part.append(X[data_point_index])
                    y_part.append(y[data_point_index])

            partitions.append((np.array(x_part), np.array(y_part)))

        return partitions

    def _partition_entropy(self, partitions, out_of_whole):
        sum = 0
        for part in partitions:
            part_label_size, part_label_result = self._count_unique(part[1])
            fraction = part_label_size / out_of_whole
            sum = sum + self._entropy(part_label_size, part_label_result) * fraction
        return sum


    def _entropy(self, out_of, values_and_counts):
        sum = 0
        for column in values_and_counts:
            for count in values_and_counts[column][1]:
                fraction = count / out_of
                sum = sum - fraction * np.log2(fraction)
        return sum

    def graph(self, class_translator=lambda x: x):
        if self.tree is None:
            return "No graph"
        return self.tree.graph(class_translator=class_translator)
    

    def __repr__(self):
        return self.graph()


