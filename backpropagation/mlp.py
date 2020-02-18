import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from tracer import SimpleTracer, ComplexTracer

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
            # print("in:" , num_send_nodes, " out:", num_recv_nodes)
            if initial_weights is not None:
                initial_inputs = np.shape(initial_weights)[0]
                if initial_inputs > num_send_nodes: 
                    # assume they know what they are doing
                    return np.array(initial_weights)
                else:
                    nodes_short = (num_send_nodes + 1) - initial_inputs
                    randos = np.random.normal((nodes_short,) + np.shape(initial_weights)[1:])
                    return np.concatenate((initial_weights, randos))
            if standard_weight is not None:
                return np.full((num_send_nodes + 1, num_recv_nodes), standard_weight)
            return np.random.uniform(-1, 1, (num_send_nodes + 1, num_recv_nodes))

        def out(self, x):
            self.tracer.nextLevel()
            self.x_aug = np.atleast_2d(x)
            x_shape = np.shape(self.x_aug)
            ones_shape = (x_shape[0],1)
            ones_array = np.ones(ones_shape)
            self.x_aug = np.concatenate((self.x_aug, ones_array), axis=1)
            self.net = np.dot(self.x_aug, self._weights)
            self.tracer.addTrace("in_data", self.x_aug)
            self.firing = self._out_func(self.net)
            self.tracer.addTrace("layer", self.serial_num).addTrace("net", self.net).addTrace("firing", self.firing)
            return self.firing

        def backProp(self, learn_rate, forward_layer, target=None):
            self.tracer.nextLevel()
            f_prime_forward = self._delta_func(self.net).reshape(-1, 1)
            self.tracer.addTrace("layer", self.serial_num).addTrace("prime forward", f_prime_forward)
            if target is not None:
                self._delta_part = (target - self.firing) * f_prime_forward
            else:
                dot_product = np.dot(forward_layer._weights, forward_layer._delta_part)[:-1]
                self._delta_part = dot_product * f_prime_forward # TODO: check if this math is right
            self.tracer.addTrace("error", self._delta_part)
            self._deltas = learn_rate * np.dot(self._delta_part, self.x_aug)
            self.tracer.addTrace("delta", self._deltas)
            return self

        def flush(self, momentum=0):
            self._weights = self._weights + np.transpose(self._deltas + momentum * self._deltas)
            self._deltas = np.zeros(np.shape(self._weights))
            self.tracer.addTrace("level", self.serial_num).addTrace("next_weights", self._weights)
            return self

        def get_weights(self):
            return self._weights

        def get_deltas(self):
            return self._deltas

        def __repr__(self):
            out = 'weights\r\n'
            out = out + str(self._weights) + '\r\ndeltas\r\n' + str(self._deltas) + '\r\n'
            return out

    def __init__(self, hidden_layer_widths, lr=.1, momentum=0, shuffle=True, deterministic=None):
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
        self.train_indicies = []
        self.test_indicies = []
        # self.tracer = SimpleTracer() # initialize simple Tracer
        self.tracer = ComplexTracer()
        self.layers = []
        self.deterministic = deterministic

    def __sigmoid__(self, net):
        return (1 + np.e ** (-1 * net)) ** -1

    def __sigmoid_prime__(self, net):
        unprime_out = self.__sigmoid__(net)
        return unprime_out * ( 1 - unprime_out)

    def _forward_pass(self, dataPoint):
        out = dataPoint
        for l in self.layers:
            out = l.out(out)
        return out

    def _backprop_and_flush(self, target, momentum=0):
        back = self.layers[-1].backProp(self.lr, None, target=target)
        for layer in reversed(self.layers[:-1]):
                back = layer.backProp(self.lr, back)
        # self.tracer.endTrace()
        for layer in self.layers:
            layer.flush(momentum)


    def fit(self, X, y, initial_weights=None, standard_weight=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

            Args:
                X (array-like): A 2D numpy array with the training data, excluding targets
                y (array-like): A 2D numpy array with the training targets
                initial_weights (array-like): allows the user to provide initial weights

            Returns:
                self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        self.initialize_weights(np.shape(X)[1], np.shape(y)[1], initial_weights, standard_weight=standard_weight)

        self.data = np.array(X)

        epochCount = 0

        while (self.deterministic is not None and epochCount < self.deterministic) or (self.deterministic is None and True):
            self._shuffle_data(self.data, y, 0)
            # for dataPoint, target in zip(self.data, y):
            for index in self.train_indicies:
                self._forward_pass(self.data[index])
                self._backprop_and_flush(y[index])
                self.tracer.nextIteration()

            print("score", self.score([self.data[x] for x in self.test_indicies], [y[x] for x in self.test_indicies]))
            print("predict", self.predict([self.data[x] for x in self.test_indicies]))


        return self

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        outs = []
        for dataPoint in X:
            outs.append(self._forward_pass(dataPoint))
        if len(outs) > 0:
            outs = np.reshape(outs, (len(outs), -1))
        return outs, np.shape(outs)

    def initialize_weights(self, inputs, outputs, initial_weights=None, standard_weight=None):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        if initial_weights is not None:
            for weights in initial_weights:
                self.layers.append(MLPClassifier.MLPWeightsLayer(self.tracer, -1, -1, self.__sigmoid__, self.__sigmoid_prime__, weights))
        else:
            prev = inputs
            for i in self.hidden_layer_widths:
                self.layers.append(MLPClassifier.MLPWeightsLayer(self.tracer, prev, i, self.__sigmoid__, self.__sigmoid_prime__, standard_weight=standard_weight))
                prev = i

            self.layers.append(MLPClassifier.MLPWeightsLayer(self.tracer, prev, outputs, self.__sigmoid__, self.__sigmoid_prime__, standard_weight=standard_weight))

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
        totals = 0
        for dataPoint, target in zip(X, y):
            out = self._forward_pass(dataPoint)
            diff = target - out
            totals = totals + (diff ** 2)
        how_many = len(y)
        return np.inf if how_many is 0 else np.sum(totals) / how_many

    def _shuffle_data(self, X, y, percent_test=0.1):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        poss_indecies = list(range(len(X)))
        self.train_indicies = []
        self.test_indicies = []
        testSize = int(percent_test * len(poss_indecies))
        if self.shuffle:
            while len(poss_indecies) > testSize:
                index = np.random.randint(0, len(poss_indecies))
                result = poss_indecies.pop(index)
                self.train_indicies.append(result)
            self.test_indicies.extend(poss_indecies)
        else:
            self.train_indicies = poss_indecies
            # one-hot attempt
            # start_test_index = np.random.randint(0, len(poss_indecies) - testSize)
            # self.train_indicies = poss_indecies[:start_test_index] + poss_indecies[start_test_index + testSize]
            # self.test_indicies = poss_indecies[start_test_index:start_test_index + testSize]

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        result = []
        for layer in self.layers:
            result.append(layer.getWeights())
        return result


    def __repr__(self):
        return str(self.layers) + '\r\n' + '-' * 20 + '\r\n' + str(self.tracer) + '\r\n'
