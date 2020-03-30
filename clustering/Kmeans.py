import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

class KMEANSClustering(BaseEstimator,ClusterMixin):

    # _base_formula_ = lambda point_x, point_y, power, rooter: sum(abs(point_x - point_y) ** power) ** rooter
    # _euclidian = lambda point_x, point_y:  KMEANSClustering._base_formula_(point_x, point_y, 2, 0.5)
    # _manhattan = lambda point_x, point_y:  KMEANSClustering._base_formula_(point_x, point_y, 1, 1)
    _euclidian = lambda point_x, point_y: np.sum(np.abs(point_x - point_y) ** 2) ** 0.5
    _manhattan = lambda point_x, point_y: np.sum(np.abs(point_x - point_y))

    def __init__(self,k=3,debug=False, distance="euclidian", window=0, tol=0): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            debug = if debug is true use the first k instances as the initial centroids otherwise choose random points as the initial centroids.
        """
        self.k = k
        self.debug = debug

        if distance == "euclidian":
            self._distance = KMEANSClustering._euclidian
        elif distance == "manhattan":
            self._distance = KMEANSClustering._manhattan
        else:
            raise SyntaxError('distance argument must be "euclidian" or "manhattan"')

        self.window = window
        self.tol = tol

    '''

    def update_closest_easy(self):
        count = 0
        for index in range(len(self.membership)):
            if self.membership[index] != self.membership_old[index]:
                count += 1
                self.centroids[self.membership_old[index]] -= self.data[index]
                self.centroids[self.membership[index]] += self.data[index]

        return count
    '''

    def _distances_from_centroids(self, points=None):
        distances = []
        points = self.data if points is None else points
        for centroid in self.centroids:
            c_dist = lambda point_y: self._distance(centroid, point_y)
            distances.append(np.apply_along_axis(c_dist, axis=1, arr=points))
        return distances

    def update_closest(self):
        distances = self._distances_from_centroids()
        new_membership = np.argmin(distances, axis=0)
        change = np.sum(self.membership != new_membership)
        self.membership = new_membership
        return change

    def update_centroids(self):
        self.centroids = np.zeros_like(self.centroids)
        counts = np.zeros((len(self.centroids), 1))
        for index in range(len(self.membership)):
            self.centroids[self.membership[index]] += self.data[index]
            counts[self.membership[index]] += 1

        for index in range(len(counts)):
            if counts[index] != 0:
                self.centroids[index] /= counts[index]
        # self.centroids /= counts       

    def fit(self,X,y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.data = X
        self.membership = np.array([None] * len(self.data))
        # self.membership_old = np.array([False] * len(self.data))
        self.centroids = []
        if self.debug:
            for i in range(self.k):
                self.centroids.append(np.copy(self.data[i]))
        else:
            indicies = np.random.randint(0, high=len(self.data), size=self.k)
            self.centroids = [np.copy(self.data[i]) for i in indicies]

        self.centroids = np.array(self.centroids)

        if abs(len(self.membership) * self.tol) < len(self.membership):
            self.tol = abs(int(len(self.membership) * self.tol))

        diff_count = np.inf
        window_count = self.window
        while window_count > -1:
            diff_count = self.update_closest()
            self.update_centroids()
            if diff_count <= self.tol:
                window_count += -1

        return self

    def _sse_clusters(self):
        distances = self._distances_from_centroids()
        sse = np.zeros((len(self.centroids),))
        counts = np.zeros((len(self.centroids),))
        for index in range(len(self.membership)):
            sse[self.membership[index]] += distances[self.membership[index]][index] ** 2
            counts[self.membership[index]] += 1
        return sse, counts


    def save_clusters(self,filename):
        """
            f = open(filename,"w+") 
            Used for grading.
            write("{:d}\n".format(k))
            write("{:.4f}\n\n".format(total SSE))
            for each cluster and centroid:
                write(np.array2string(centroid,precision=4,separator=","))
                write("\n")
                write("{:d}\n".format(size of cluster))
                write("{:.4f}\n\n".format(SSE of cluster))
            f.close()
        """
        f = None
        try:
            f = open(filename, "w+")
            f.write("{:d}\n".format(self.k))
            sse_for_clusters, counts = self._sse_clusters()
            f.write("{:.4f}\n\n".format(np.sum(sse_for_clusters)))
            for centroid, sse, count in zip(self.centroids, sse_for_clusters, counts):
                f.write(np.array2string(centroid,precision=4,separator=","))
                f.write("\n")
                f.write("{:d}\n".format(int(count)))
                f.write("{:.4f}\n\n".format(sse))
        finally:
            if f != None:
                f.close()

    def __repr__(self):
        out = ''
        for index in range(len(self.centroids)):
            out += 'centroid:' + str(self.centroids[index]) + '\n'
            out += 'members:\n'
            for member_index in range(len(self.data)):
                if index == self.membership[member_index]:
                    out += str(self.data[member_index]) + '\n'
            out += '*********************\n'
        return out
