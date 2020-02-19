import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class CalcedValue:
    def __init__(self, unlocked_value=None):
        self.locked = False
        self.value = unlocked_value


class Node:
    serial = 0
    def __init__(self, out_value=None):
        self.out_value = CalcedValue(0)
        self.serial_num = Node.serial
        Node.serial = Node.serial + 1

    def set_value(self, value):
        self.out_value.value = value

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
        self.out_value.locked = False

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

    def set_prev_nodes(self, prev_nodes):
        self.prev_nodes = prev_nodes

    def set_prev_weights(self, prev_weights):
        self.prev_weights = prev_weights

    def get_prev_weights(self):
        return self.prev_weights

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

    def __sigmoid__(self, net):
        return (1 + np.e ** (-1 * net)) ** -1

    def __sigmoid_prime__(self, net):
        unprime_out = self.__sigmoid__(net)
        return unprime_out * (1 - unprime_out)


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
        self._make_hidden_layers()
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle

        self.layers = []

    def _make_hidden_layers(self):
        self.h_layers = [None]
        for w in self.hidden_layer_widths:
            curr = [InnerNode(self.h_layers[-1], None, self.__sigmoid__, self.__sigmoid_prime__)] * w
            curr.append(Node(1)) # bias node
            self.h_layers.append(curr)
        self.h_layers = self.h_layers[1:]
    
    def _make_input_layer(self, width):
        self.in_layer = [Node()] * width
        self.in_layer.append(Node(1)) # bias node

    def _make_output_layer(self, width):
        self.out_layer = [OuterNode(self.h_layers[-1], None, self.__sigmoid__, self.__sigmoid_prime__)] * width

    def fit(self, X, y, initial_weights=None, initial_standard=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.data = np.atleast_2d(X)
        self.targets = np.atleast_2d(y)
        x_shape = np.shape(self.data)
        self._make_input_layer(x_shape[1])
        y_shape = np.shape(self.targets)
        self._make_output_layer(y_shape[0])
        for node in self.h_layers[0][:-1]: # last hook-up
            node.set_prev_nodes(self.in_layer)

        
        self.initialize_weights(initial_weights, initial_standard)

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

    def initialize_weights(self, initial_weights, initial_standard):
        """ Initialize weights for perceptron. Don't forget the bias!
        Returns:
        """
        if initial_weights is not None:
            # they know what they are doing
            for index, weights in zip(self.h_layers, initial_weights):
                for nodeIndex in range(len(self.h_layers[index])):
                    # slice the column for the node
                    self.h_layers[index][nodeIndex].set_prev_weights(weights[:,nodeIndex])
            for nodeIndex in range(len(self.out_layer)):
                # and the initial weights out to the output layer
                self.out_layer[nodeIndex].set_prev_weights(initial_weights[-1][:,nodeIndex])
        elif initial_standard is not None:
            # weights = []
            # first = np.full((np.shape(self.in_layer)[0], np.shape(self.h_layers[0])[0]), initial_standard)
            # weights.append(first)
            # for layerIndex in range(len(self.h_layers)):
            #     weights.append(np.full((np.shape(self.h_layers))))


            for node in self.h_layers[0]:
                node.set_prev_weights([initial_standard] * np.shape(self.in_layer)[0])
            for layerIndex in range(len(self.h_layers[0:])):
                for node in self.h_layers[layerIndex + 1]:
                    node.set_prev_weights([initial_standard] * np.shape(self.h_layers[layerIndex])[0])
            for node in self.out_layer:
                node.set_prev_weights([initial_standard] * np.shape(self.h_layers[-1])[0])
        else:
            for node in self.h_layers[0]:
                node.set_prev_weights(np.random.normal(size=np.shape(self.in_layer))
            for layerIndex in range(len(self.h_layers[0:])):
                for node in self.h_layers[layerIndex + 1]: # to account for the slice
                    node.set_prev_weights(np.random.normal(size=np.shape(self.h_layers[layerIndex])))
            for node in self.out_layer:
                node.set_prev_weights(np.random.normal(size=np.shape(self.h_layers[-1])))



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
            
        