import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from tracer import SimpleTracer

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

class MLPClassifier(BaseEstimator,ClassifierMixin):

    class MLPWeightsLayer:
        serial = 0

        '''
                   o1 o2
                f1 #  #
                f2 #  #
                B  #  #
        '''

        def __init__(self, tracer, num_send_nodes, num_recv_nodes, output_func, delta_func, initial_weights=None, standard_weight=None):
            self._weights = self.__initialize_weights__(num_send_nodes, num_recv_nodes, initial_weights, standard_weight)
            self._deltas = np.zeros(np.shape(self._weights))
            self._out_func = output_func
            self._delta_func = delta_func
            self.tracer = tracer
            self.serial_num = MLPClassifier.MLPWeightsLayer.serial
            MLPClassifier.MLPWeightsLayer.serial = MLPClassifier.MLPWeightsLayer.serial + 1

        def __initialize_weights__(self, num_send_nodes, num_recv_nodes, initial_weights=None, standard_weight=None):
            print("in:" , num_send_nodes, " out:", num_recv_nodes)
            if initial_weights != None:
                return np.concatenate(initial_weights, np.random.normal()) # TODO: CHECK THE SIZE for the random
            if standard_weight != None:
                return np.full((num_send_nodes + 1, num_recv_nodes), standard_weight)
            return np.random.uniform(-1, 1, (num_send_nodes + 1, num_recv_nodes))

        def out(self, x):
            x_shape = np.shape(x)
            ones_shape = (1,) +  x_shape[1:]
            ones_array = np.ones(ones_shape)
            print("precat", x, x_shape, ones_shape, ones_array)
            x_aug = np.concatenate((x, ones_array))
            print("out", x_aug, "\n\r", self._weights)
            net = np.dot(x_aug, self._weights)
            firing = self._out_func(net)
            self.tracer.addTrace("layer", self.serial_num).addTrace("net", net).addTrace("firing", firing)
            return firing

        def backProp(self, learn_rate, forward_layer, target=None):
            pass

        def flush(self):
            self._weights = self._weights + self._deltas
            self._deltas = np.zeros(np.shape(self._weights))

        def get_weights(self):
            return self._weights

        def get_deltas(self):
            return self._deltas

        def __repr__(self):
            out = 'weights\r\n'
            out = out + str(self._weights) + '\r\ndeltas\r\n' + str(self._deltas) + '\r\n'
            return out

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
        self.tracer = SimpleTracer() # initialize simple Tracer
        self.layers = []

    def __sigmoid__(self, net):
        return (1 + np.e ** (-1 * net)) ** -1

    def __sigmoid_prime__(self, net):
        return self.__sigmoid__(net) * ( 1 - self.__sigmoid__(net))


    def fit(self, X, y, initial_weights=None, standard_weight=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        print("shapes", np.shape(X), np.shape(y))
        self.initialize_weights(np.shape(X)[1], np.shape(y)[1], initial_weights, standard_weight=standard_weight)

        # self.data = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        self.data = np.array(X)
        print("data\r\n", self.data)
        # target = y[0]
        print("layers\r\n", self.layers)
        for dataPoint in self.data:
            out = dataPoint
            for l in self.layers:
                out = l.out(out)


        self.tracer.endTrace()
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

    def initialize_weights(self, inputs, outputs, initial_weights=None, standard_weight=None):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        if initial_weights:
            for w in initial_weights:
                self.layers.append(MLPClassifier.MLPWeightsLayer(self.tracer, -1, -1, self.__sigmoid__, lambda x: x, w))
        else:
            prev = inputs
            for i in self.hidden_layer_widths:
                self.layers.append(MLPClassifier.MLPWeightsLayer(self.tracer, prev, i, self.__sigmoid__, lambda x: x, standard_weight=standard_weight))
                prev = i

            self.layers.append(MLPClassifier.MLPWeightsLayer(self.tracer, prev, outputs, self.__sigmoid__, lambda x: x, standard_weight=standard_weight))

        # return [0]

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


    def __repr__(self):
        return str(self.layers) + '\r\n' + '-' * 20 + '\r\n' + str(self.tracer) + '\r\n'
