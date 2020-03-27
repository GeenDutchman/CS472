import numpy as np
from HAC import HACClustering as Hac

def basic_hac():
    a = np.array([[.8, .7], [0,0], [1,1],[4,4]])

    hac = Hac(link_type='complete', distance='manhattan', k=3)
    print(hac.fit(a))
    hac.save_clusters('try.txt')

basic_hac()