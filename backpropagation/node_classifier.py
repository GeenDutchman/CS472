import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class CalcedValue:
    def __init__(self, unlocked_value=None):
        self.locked = False
        self.value = unlocked_value


class Node:
    serial = 0
    def __init__(self, out_value):
        self.out_value = CalcedValue(0)
        self.serial_num = Node.serial
        Node.serial = Node.serial + 1

    def get_net_value(self):
        return self.out_value.value

    def get_out_value(self):
        return self.out_value.value

    def take_blame(self, serial_number, send_blame):
        pass

    def get_error(self):
        return 0

    def push_blame(self):
        pass

    def get_deltas(self, learn_rate):
        return []

    def flush(self, learn_rate, momentum):
        pass

class InnerNode(Node):
    def __init__(self, prev_nodes, prev_weights, out_func, back_func):
        super().__init__(0)
        self.prev_nodes = prev_nodes
        self.prev_weights = prev_weights
        self.out_func = out_func
        self.back_func = back_func
        self.net = CalcedValue()
        self.error = CalcedValue()
        self.deltas = CalcedValue()
        self.received_blame = {}


    def get_net_value(self):
        if not self.net.locked:
            self.net.value = 0
            for node, weight in zip(self.prev_nodes, self.prev_weights):
                self.net.value = self.net.value + node.get_out_value() * weight
            self.net.locked = True
        return self.net.value

    def get_out_value(self):
        if not self.out_value.locked:
            self.out_value = self.out_func(self.get_net_value())
            self.out_value.locked = True
        return self.out_value.value

    def take_blame(self, serial_number, send_blame):
        self.received_blame[serial_number] = send_blame

    def get_error(self):
        if not self.error.locked:
            self.error.value = 0
            for node in self.received_blame:
                self.error.value = self.error.value + self.received_blame[node]
            self.error.value = self.error.value * self.back_func(self.get_net_value())
            self.error.locked = True
        return self.error.value

    def push_blame(self):
        error = self.get_error()
        for node, weight in zip(self.prev_nodes, self.prev_weights):
            node.take_blame(self.serial_num, error * weight)

    def get_deltas(self, learn_rate):
        if not self.deltas.locked:
            error = self.get_error()
            self.deltas.value = []
            for node in self.prev_nodes:
                self.deltas.value.append(learn_rate * node.get_out_value() * error)
            self.deltas.locked = True
        return self.deltas.value

    def flush(self, learn_rate, momentum):
        deltas = self.get_deltas(learn_rate)
        for index in range(len(self.prev_weights)):
            self.prev_weights[index] = self.prev_weights[index] + deltas[index] + momentum * deltas[index]
        
        # do some resets
        self.deltas.locked = False
        self.error.locked = False
        self.net.locked = False
        self.out_value = False
        self.received_blame = {}

class OuterNode(InnerNode):
    def __init__(self, prev_nodes, prev_weights, out_func, back_func):
        super().__init__(prev_nodes, prev_weights, out_func, back_func)





### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, hidden_layer_widths, lr=.1, momentum=0, shuffle=True):
        """ Initialize class with chosen hyperparameters.
        Args:
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        Example:
            mlp = MLPClassifier([3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle

        self.layers = []
        for width in 


    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.initial_weights = self.initialize_weights() if not initial_weights else initial_weights

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

    def initialize_weights(self):
        """ Initialize weights for perceptron. Don't forget the bias!
        Returns:
        """

        return [0]

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

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        pass

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        pass
            
        