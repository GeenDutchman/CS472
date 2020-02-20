import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from tracer import SimpleTracer, ComplexTracer

# NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.


class MLPClassifier(BaseEstimator, ClassifierMixin):


    class MLPWeightsLayer:
        serial = 0

        '''
                   o1 o2
                f1 #  #
                f2 #  #
                B  #  #
        '''

        def __init__(self, tracer, num_send_nodes, num_recv_nodes, output_func, delta_func, initial_weights=None, standard_weight=None):
            self._weights = self.__initialize_weights__(
                num_send_nodes, num_recv_nodes, initial_weights, standard_weight)
            self._deltas = np.zeros(np.shape(self._weights))
            self._old_deltas = np.copy(self._deltas)
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
                    randos = np.random.normal(
                        (nodes_short,) + np.shape(initial_weights)[1:])
                    return np.concatenate((initial_weights, randos))
            if standard_weight is not None:
                return np.full((num_send_nodes + 1, num_recv_nodes), standard_weight)
            return np.random.uniform(-1, 1, (num_send_nodes + 1, num_recv_nodes))

        def out(self, x):
            self.tracer.nextLevel()
            self.x_aug = np.atleast_2d(x)
            x_shape = np.shape(self.x_aug)
            ones_shape = (x_shape[0], 1)
            ones_array = np.ones(ones_shape)
            self.x_aug = np.concatenate((self.x_aug, ones_array), axis=1)
            self.tracer.addTrace("in_data", self.x_aug).addTrace("weights", self._weights)
            self.net = np.dot(self.x_aug, self._weights)
            self.firing = self._out_func(self.net)
            self.tracer.addTrace("layer", self.serial_num).addTrace("net", self.net).addTrace("firing", self.firing)
            return self.firing

        def backProp(self, learn_rate, forward_layer, target=None):
            self.tracer.nextLevel()
            f_prime_forward = self._delta_func(self.net)
            self.tracer.addTrace("layer", self.serial_num).addTrace("prime forward", f_prime_forward)
            if target is not None:
                self._delta_part = (target - self.firing) * f_prime_forward
            else:
                dot_product = np.dot(forward_layer._delta_part, np.transpose(forward_layer._weights))
                # TODO: check if this math is right
                self._delta_part = np.multiply(f_prime_forward, dot_product[:,:-1])
            self.tracer.addTrace("error", self._delta_part)
            # self._old_deltas = np.copy(self._deltas)
            self._deltas = learn_rate * np.dot(np.transpose(self.x_aug), self._delta_part)
            return self

        def flush(self, momentum=0):
            momentum_delta = momentum * self._old_deltas
            self._deltas = self._deltas + momentum_delta
            self.tracer.addTrace("delta", self._deltas)
            self._weights = self._weights + self._deltas
            self._old_deltas = np.copy(self._deltas)
            self.tracer.addTrace("level", self.serial_num).addTrace("next_weights", self._weights)
            return self

        def get_weights(self):
            return self._weights

        def get_deltas(self):
            return self._deltas

        def __repr__(self):
            out = 'weights\r\n'
            out = out + str(self._weights) + '\r\ndeltas\r\n' + \
                str(self._deltas) + '\r\n'
            return out

        def csv_print(self):
            out = ''
            for in_node in self._weights:
                for weight in in_node:
                    out = out + str(weight) + ',\n'
            return out

                

    def __init__(self, hidden_layer_widths, lr=.1, momentum=0, shuffle=True, deterministic=None, one_hot=False):
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
        self.verify_indicies = []
        # self.tracer = SimpleTracer() # initialize simple Tracer
        self.tracer = ComplexTracer().setActive(False)
        self.layers = []
        self.deterministic = deterministic
        # for debugging
        self.print_last_not_all_trace = False
        self.one_hot = one_hot


    def __sigmoid__(self, net):
        return (1 + np.e ** (-1 * net)) ** -1

    def __sigmoid_prime__(self, net):
        unprime_out = self.__sigmoid__(net)
        return unprime_out * (1 - unprime_out)

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

    def fit(self, X, y, initial_weights=None, standard_weight=None, percent_verify=0.1, window=4, momentum=0, tolerance=1e-3):
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

        self.epochCount = 0
        self.bssf = [np.inf, self.get_weights()]

        # pre-shuffle and split into train and verify sets
        self._train_validate_split(self.data, y, percent_verify)
        self._shuffle_data()
        # see how good it is initially
        score = self._calc_l2_err([self.data[x] for x in self.verify_indicies], [y[x] for x in self.verify_indicies])

        window_countdown = window

        while (self.deterministic is not None and self.epochCount < self.deterministic) or (self.deterministic is None and window_countdown > 0):
            # for dataPoint, target in zip(self.data, y):
            self.tracer.addTrace("epoch", self.epochCount)
            for index in self.train_indicies:
                self._forward_pass(self.data[index])
                self._backprop_and_flush(y[index], momentum)

            score = self._calc_l2_err([self.data[x] for x in self.verify_indicies], [
                               y[x] for x in self.verify_indicies])
            self.tracer.addTrace("score", score)

            # if infinite or big enough change, commit it
            score_diff = self.bssf[0] - score
            if self.bssf[0] is np.inf or (score_diff > 0 and abs(score_diff) > tolerance):
                self.bssf = [score, self.get_weights()]
                window_countdown = window
            else:
                window_countdown = window_countdown - 1

            self._shuffle_data()
            self.epochCount = self.epochCount + 1
            self.tracer.endTrace()
            self.tracer.nextIteration()

        # print("bssf\n", bssf)
        self.initialize_weights(-1, -1, initial_weights=self.bssf[1])# TODO: uncomment me!

        return self

    def getEpochCount(self):
        return self.epochCount

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
        outs = np.array(outs)
        return outs, np.shape(outs)

    def initialize_weights(self, inputs, outputs, initial_weights=None, standard_weight=None):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        if initial_weights is not None:
            self.layers = []
            for weights in initial_weights:
                self.layers.append(MLPClassifier.MLPWeightsLayer(
                    self.tracer, -1, -1, self.__sigmoid__, self.__sigmoid_prime__, weights))
        else:
            prev = inputs
            for i in self.hidden_layer_widths:
                self.layers.append(MLPClassifier.MLPWeightsLayer(
                    self.tracer, prev, i, self.__sigmoid__, self.__sigmoid_prime__, standard_weight=standard_weight))
                prev = i

            self.layers.append(MLPClassifier.MLPWeightsLayer(self.tracer, prev, outputs,
                                                             self.__sigmoid__, self.__sigmoid_prime__, standard_weight=standard_weight))

        # return [0]

    def _calc_l2_err(self, X, y):
        outs = self.predict(X)[0]
        totals = 0
        for out, target in zip(outs, y):
            diff = target - out
            totals = totals + np.sum(diff ** 2)
        how_many = len(y)
        # assume that if there is no validation data, anything will improve it
        return np.inf if how_many is 0 else np.sum(totals) / how_many

    def _do_one_hot(self, y, depress=False):
        if not self.one_hot:
            return y
        result = y
        max_val_index = np.argmax(y)
        for index in range(len(result)):
            if index == max_val_index:
                result[index] = 1
            elif depress:
                result[index] = 0
        return result

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

            Args:
                X (array-like): A 2D numpy array with data, excluding targets
                y (array-like): A 2D numpy array with targets

            Returns:
                score : float
                    Mean accuracy of self.predict(X) wrt. y.
        """
        outs = self.predict(X)[0]
        correct = 0
        if self.one_hot:
            outs = [self._do_one_hot(out, depress=True) for out in outs]
        for out, target in zip(outs, y):
            if np.array_equal(out, target):
                correct = correct + 1
        return correct / len(y)

    def _train_validate_split(self, X, y, percent_verify=0.1):
        poss_indecies = list(range(len(X)))

        self.train_indicies = []
        self.verify_indicies = []
        testSize = int(percent_verify * len(poss_indecies))
        # if testSize <= 0:
        #     testSize = 1
        # split for validaiton
        start_test_index = np.random.randint(
            0, len(poss_indecies) - testSize)
        self.train_indicies = poss_indecies[:start_test_index] + poss_indecies[start_test_index + testSize:]
        self.verify_indicies = poss_indecies[start_test_index:start_test_index + testSize]       


    def _shuffle_data(self):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        if self.shuffle:
            to_shuffle = self.train_indicies
            leave_how_many = 0
            self.train_indicies = []
            while len(to_shuffle) > leave_how_many:
                index = np.random.randint(0, len(to_shuffle))
                result = to_shuffle.pop(index)
                self.train_indicies.append(result)
        

    # Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        result = []
        for layer in self.layers:
            result.append(layer.get_weights())
        return result

    def __repr__(self):
        # return str(self.layers) + '\r\n' + '-' * 20 + '\r\n' + (self.tracer.iteration_to_string(-1) if self.print_last_not_all_trace else str(self.tracer)) + '\r\n'
        return str(self.tracer) + "\r\n"

    def csv_print(self):
        out = ''
        for layer in self.layers:
            out = out + layer.csv_print()
        return out
