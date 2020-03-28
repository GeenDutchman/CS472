import numpy as np
from HAC import HACClustering as Hac
from Kmeans import KMEANSClustering as Kmeans

def basic_hac():
    a = np.array([[.8, .7], [0,0], [1,1],[4,4]])

    hac = Hac(link_type='complete', distance='manhattan', k=3)
    print(hac.fit(a))
    hac.save_clusters('try_hac.txt')

def basic_kmeans():
    a = np.array([[.9, .8], [.2, .2], [.7,.6], [-.1, -.6], [.5, .5]])

    kmeans = Kmeans(k=2, debug=True, window=1)
    print(kmeans.fit(a))
    kmeans.save_clusters('try_kmeans.txt')

# basic_hac()
basic_kmeans()