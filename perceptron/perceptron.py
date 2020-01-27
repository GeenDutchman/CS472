import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

from sklearn.linear_model import Perceptron

class PerceptronClassifier(BaseEstimator,ClassifierMixin):

    class PerceptronTrace:
        def __init__(self):
            self.container = {}
            self.count = 0

        def addTrace(self, key, result):
            if key not in self.container:
                self.container[key] = []
            self.container[key].append(result)
            return self

        def getElement(self, key, count=-1):
            if count == -1:
                count = self.count
            return self.container[key][count]

        def getColumns(self, *keys):
            results = []
            for c in range(self.count - 1):
                results.append([])
                for k in keys:
                    results[c].append(self.container[k][c])
            return np.array(results)




        def getCurrentCount(self):
            return self.count

        def endTrace(self):
            for key in self.container:
                self.container[key].append('-')
            self.count = self.count + 1
            return self

        def nextLevel(self):
            self.count = self.count + 1
        
        def __repr__(self):
            out = ''
            for key in self.container:
                # print(key, end='\t')
                out = out + str(key) + '\t'
            # print()
            out = out + "\n\r"
            for i in range(self.count):
                for key in self.container:
                    out = out + str(self.container[key][i]) + '\t'
                    # print(self.container[key][i], end='\t')
                out = out + "\n\r"
                # print()
            # print()
            out = out + "\n\r"
            return out

    def __init__(self, lr=.1, shuffle=True, activationFunction = lambda activation: 1 if activation > 0 else 0, printIt=True):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        """
        self.lr = lr
        self.shuffle = shuffle
        self.activationFunction = activationFunction
        self.printIt = printIt
        self.trainTrace = PerceptronClassifier.PerceptronTrace()
        self.executeTrace = PerceptronClassifier.PerceptronTrace()

    def __repr__(self):
        return str(self.trainTrace) + '\r\n' + str(self.executeTrace) + '\r\n'

    def _pcPrint(self, *values: object, sep: str=' ', end: str='\n'):
        # method header copied from builtins
        if self.printIt:
            print(*values, sep=sep, end=end)


    def _for_data_point(self, x, tracer):
        activation = np.dot(x, self.weights)
        firing = self.activationFunction(activation)
        # self._pcPrint(activation, firing, end='\t', sep='\t')
        tracer.addTrace("activation", activation).addTrace("firing", firing)
        return firing

    def _add_bias_node(self, X):
        # augment data with bias node
        inData = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        # self._pcPrint("inData", inData)
        return inData

    def _stopper(self, tracer, tol=0.0005):
        numDataPoints = len(self.inData) -1
        count = tracer.getCurrentCount() -1 
        l1Loss = 0

        for i in range(count, count - numDataPoints, -1):
            difference = tracer.getElement("difference", i)
            l1Loss = l1Loss + (difference if difference > 0 else difference * -1)

        self._pcPrint("l1Loss", l1Loss)
        if l1Loss < tol:
            return True
        return False
        


    def _stochastic(self, tracer):
        for dataPoint, target in zip(self.inData, self.targetData):
            # self._pcPrint(dataPoint, target, np.transpose(self.weights), end='\t', sep='\t')
            tracer.addTrace("dataPoint", dataPoint).addTrace("target", target).addTrace("weights", np.transpose(self.weights))
            firing = self._for_data_point(dataPoint, tracer)
            difference = target - firing
            tracer.addTrace("difference", difference)
            deltas = self.lr * (difference) * dataPoint
            # self._pcPrint(firing, deltas, sep='\t')
            tracer.addTrace("deltas", deltas)
            self.weights += np.reshape(deltas, (-1, 1))
            tracer.nextLevel()

    # def _batch(self):
    #     activations = np.dot(self.inData, self.weights)
    #     firing = self.activationFunction(activations)
    #     # self._pcPrint('firing', firing)
    #     self.weights += self.lr * np.dot(np.transpose(self.inData), activations - self.targetData)

    def fit(self, X, y, initial_weights=None, epochs=1):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        self.inData = self._add_bias_node(X)
        self.targetData = y

        need_initialize_weights = True if ((initial_weights is None) or (initial_weights.size is 0)) else False
        self.initial_weights = self.initialize_weights(0) if need_initialize_weights else initial_weights

        self.weights = self.initial_weights

        for dummyEpoch in range(epochs):
            self._stochastic(self.trainTrace)
            if self._stopper(self.trainTrace):
                break

        self.trainTrace.endTrace()
        self._pcPrint(self.trainTrace)
        return self

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        augmented = self._add_bias_node(X)
        for dataPoint in augmented:
            self._for_data_point(dataPoint, self.executeTrace)
            self.executeTrace.nextLevel()
        self.executeTrace.endTrace()
        self._pcPrint(self.executeTrace)
        self._pcPrint(self.executeTrace.getColumns("firing"))
        firingResults = self.executeTrace.getColumns("firing")
        return firingResults, firingResults.shape
        

    def initialize_weights(self, standard_weight_value=None):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        features = np.shape(self.inData)[0] # count of features
        self._pcPrint("features",  features)
        outputs = np.shape(self.targetData)[1] # how many dimensions of outputs?? #TODO figure this out

        w = None
        if standard_weight_value != None:
            w = np.full((features, outputs), standard_weight_value * 1.0)
        else:
            w = np.random.rand(features, outputs)
        # self.initial_weights = w
        return w

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
        return self.weights
