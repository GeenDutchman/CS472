import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from triangle_table import TriangleTable

class HACClustering(BaseEstimator,ClusterMixin):

    # _base_formula_ = lambda point_x, point_y, power, rooter: sum(abs(point_x - point_y) ** power) ** rooter
    # _euclidian = lambda point_x, point_y:  HACClustering._base_formula_(point_x, point_y, 2, 0.5)
    # _manhattan = lambda point_x, point_y:  HACClustering._base_formula_(point_x, point_y, 1, 1)
    _euclidian = lambda point_x, point_y: np.sum(np.abs(point_x - point_y) ** 2) ** 0.5
    _manhattan = lambda point_x, point_y: np.sum(np.abs(point_x - point_y))

    def __init__(self,k=3,link_type='single',distance="euclidian"): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        self.link_type = link_type
        self.k = k
        self.distance_table = TriangleTable()

        if distance == "euclidian":
            self._distance = HACClustering._euclidian
        elif distance == "manhattan":
            self._distance = HACClustering._manhattan
        else:
            raise SyntaxError('distance argument must be "euclidian" or "manhattan"')

    def _calc_dist(self, group_a, group_b):
        close_far = [np.inf, -np.inf]
        for a_index in group_a:
            for b_index in group_b:
                dist = self._distance(self.data[a_index], self.data[b_index])
                if dist < close_far[0]:
                    close_far[0] = dist
                if dist > close_far[1]:
                    close_far[1] = dist
        if self.link_type == 'complete':
            return close_far[1]
        else:
            return close_far[0]

    def _dist(self, group_a, group_b):
        dist = np.inf
        try:
            dist = self.distance_table.get(group_a, group_b)
        except KeyError:
            dist = self._calc_dist(group_a, group_b)
            self.distance_table.add(group_a, group_b, dist)

        return dist

        

    def _find_closest(self):
        closest = (np.inf, []) # dist, close_groups
        latest_level = self.levels[-1]
        for group_a_index in range(len(self.tree[latest_level])):
            for group_b_index in range(group_a_index + 1, len(self.tree[latest_level])):
                group_a = self.tree[latest_level][group_a_index]
                group_b = self.tree[latest_level][group_b_index]
                dist = self._dist(group_a, group_b)
                if dist < closest[0]:
                    closest = (dist, (group_a, group_b))
        return closest

    def fit(self,X,y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.tree = dict()
        self.levels = []
        self.data = X

        self.levels.append((0,0)) # index, dist

        self.tree[self.levels[-1]] = []
        for i in range(len(self.data)):
            self.tree[self.levels[-1]].append((i,))

        while(len(self.tree[self.levels[-1]]) > 1):
            closest = self._find_closest() # closest = (dist, close_groups)
            next_level = (len(self.levels), closest[0])
            self.tree[next_level] = []
            for group in self.tree[self.levels[-1]]:
                if group in closest[1]:
                    continue
                self.tree[next_level].append(group)
            new_group = sum(closest[1], tuple())
            self.tree[next_level].append(new_group)
            self.levels.append(next_level)

        return self

    def cluster_SSE(self, cluster):
        total = np.zeros(np.shape(self.data[cluster[0]]))
        for index in cluster:
            total = total + self.data[index]
        centroid = total / len(cluster)
        sse = 0
        for index in cluster:
            sse = sse + (self._distance(self.data[index], centroid) ** 2)

        str_out = np.array2string(centroid,precision=4,separator=',') + '\n'
        str_out = str_out + "{:d}".format(len(cluster)) + '\n'
        str_out = str_out + "{:.4f}".format(sse) + '\n\n'
        return centroid, sse, str_out

    def key_SSE(self, key=None):
        key = key if key != None else self.levels[-1]
        str_out = ''
        sse_total = 0
        for index in range(len(self.tree[key])):
            centroid, sse, cluster_str_out = self.cluster_SSE(self.tree[key][index])
            str_out += cluster_str_out
            sse_total += sse
        
        return sse_total, "{:.4f}\n\n".format(sse_total) + str_out
   

    def save_clusters(self,filename, key=None):
        """
            f = open(filename,"w+") 
            #Used for grading.
            write("{:d}\n".format(k))
            write("{:.4f}\n\n".format(total SSE))
            for each cluster and centroid:
                write(np.array2string(centroid,precision=4,separator=","))
                write("\n")
                write("{:d}\n".format(size of cluster))
                write("{:.4f}\n\n".format(SSE of cluster))
            f.close()
        """
        key = -1 * self.k if key == None else -1
        f = None
        try:
            f = open(filename, 'w+')
            f.write("{:d}\n".format(self.levels[key][1]))
            last_group = self.levels[-1 * abs(key)]
            level_result = self.key_SSE(last_group)
            f.write(level_result[1])
        finally:
            if f != None:
                f.close()
        

    def __repr__(self):
        out = ''
        for index in reversed(self.levels):
            out = out + 'distance:' + str(index[1]) + ' number of groups:' + str(len(self.tree[index])) + ' groups:' + str(self.tree[index])
            out = out + '\n'
        return out


