import numpy as np
from HAC import HACClustering as Hac

def basic_hac():
    a = np.array([[.8, .7], [0,0], [1,1],[4,4]])

    hac = Hac(link_type='complete', distance='manhattan', k=1)
    print(hac.fit(a).distance_table)

basic_hac()