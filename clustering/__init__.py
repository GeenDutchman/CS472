import numpy as np
from sklearn.preprocessing import MinMaxScaler

from arff import Arff
from HAC import HACClustering as Hac
from Kmeans import KMEANSClustering as Kmeans

def basic_hac():
    a = np.array([[.8, .7], [0,0], [1,1],[4,4]])

    # hac = Hac(link_type='complete', distance='manhattan', k=3)
    # print(hac.fit(a))
    # hac.save_clusters('try_hac.txt')

    hac2 = Hac(link_type='single', distance='manhattan', k=3)
    print(hac2.fit(a))
    hac2.save_clusters('try_hac.txt')

def basic_kmeans():
    a = np.array([[.9, .8], [.2, .2], [.7,.6], [-.1, -.6], [.5, .5]])

    kmeans = Kmeans(k=2, debug=True, window=1)
    print(kmeans.fit(a))
    kmeans.save_clusters('try_kmeans.txt')

def debug():
    mat = Arff("./data/abalone.arff",label_count=0) ## label_count = 0 because clustering is unsupervised.

    raw_data = mat.data
    data = raw_data

    scaler = MinMaxScaler()
    scaler.fit(data)
    norm_data = scaler.transform(data)

    kmeans = Kmeans(k=5, debug=True)
    kmeans.fit(norm_data)
    kmeans.save_clusters('debug_kmeans.txt')

    hac_single = Hac(link_type="single", k=5)
    hac_single.fit(norm_data)
    hac_single.save_clusters('debug_hac_single.txt')

    hac_complete = Hac(link_type='complete', k=5)
    hac_complete.fit(norm_data)
    hac_complete.save_clusters('debug_hac_complete.txt')



# basic_hac()
# basic_kmeans()
debug()