import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans as Sk_Kmeans
from sklearn.cluster import AgglomerativeClustering as Sk_HAC

from arff import Arff
from HAC import HACClustering as Hac
from Kmeans import KMEANSClustering as Kmeans

_base_formula_ = lambda point_x, point_y, power, rooter: sum(abs(point_x - point_y) ** power) ** rooter
_euclidian = lambda point_x, point_y:  HACClustering._base_formula_(point_x, point_y, 2, 0.5)

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


def evaluation():
    mat = Arff("./data/seismic-bumps_train.arff",label_count=0) ## label_count = 0 because clustering is unsupervised.

    raw_data = mat.data
    data = raw_data

    scaler = MinMaxScaler()
    scaler.fit(data)
    norm_data = scaler.transform(data)

    kmeans = Kmeans(k=5, debug=True)
    kmeans.fit(norm_data)
    kmeans.save_clusters('evaluation_kmeans.txt')

    hac_single = Hac(link_type="single", k=5)
    hac_single.fit(norm_data)
    hac_single.save_clusters('evaluation_hac_single.txt')

    hac_complete = Hac(link_type='complete', k=5)
    hac_complete.fit(norm_data)
    hac_complete.save_clusters('evaluation_hac_complete.txt')

def iris():
    mat = Arff("./data/iris.arff", label_count=0)
    ldata = mat.data
    nldata = ldata[:-1]

    hac_single_l = Hac(link_type="single", k=k)
    hac_single.fit(ldata)
    print("single calculated")

    hac_complete_l = Hac(link_type="complete", k=k)
    hac_complete.fit(ldata)
    print("complete calculated")

    hac_single_nl = Hac(link_type="single", k=k)
    hac_single.fit(nldata)
    print("nl single calculated")

    hac_complete_nl = Hac(link_type="complete", k=k)
    hac_complete.fit(nldata)
    print("nl complete calculated")

    def run_both(k, dataset):
        # sse_set[0].append(k)
        print("k =", k)

        kmeans = Kmeans(k=k, debug=False)
        kmeans.fit(dataset)
        kmeans.save_clusters("./out/" + str(k) + "_iris_kmeans.txt")
        # sse_set[1].append(kmeans._sse_clusters()[0])


        hac_single_l.save_clusters("./out/" + str(k) + "_hac_single_l.txt")
        # sse_set[2].append(hac_single.key_SSE()[0])

        
        hac_complete_l.save_clusters("./out/" + str(k) "_hac_complete_l.txt")
        # sse_set[3].append(hac_single.key_SSE()[0])

        hac_single_nl.save_clusters("./out/" + str(k) + "_hac_single_nl.txt")
        # sse_set[2].append(hac_single.key_SSE()[0])

        
        hac_complete_nl.save_clusters("./out/" + str(k) + "_hac_complete_nl.txt")
        # sse_set[3].append(hac_single.key_SSE()[0])


    # sse_set_no_label = [[],[],[],[]]
    # sse_with_label = [[],[],[],[]]

    for k in range(2, 8):
        run_both(k, nldata)
        run_both(k, ldata)

    # np.savetxt('./out/iris_no_label.txt', sse_set_no_label, delimiter=',')
    # np.savetxt('./out/iris_with_label.txt', sse_with_label, delimiter=',')

    # kmeans_4 = []

    for dummy_iteration in range(5):
        print("dummy_iteration", dummy_iteration)
        kmeans = Kmeans(k=4, debug=False)
        kmeans.fit(ldata)
        # kmeans_4.append(kmeans._sse_clusters()[0])
        kmeans.save_clusters("./out/" + str(k) + "_kmeans_4.txt")

    # np.savetxt('./out/kmeans_4.txt', kmeans_4, delimiter=',')

def sk_iris():
    mat = Arff("./data/iris.arff", label_count=0)
    ldata = mat.data

    def run_both(k, dataset, sse_set):
        sse_set[0].append(k)

        kmeans = Sk_Kmeans(5, init='random', n_init=1)
        kmeans.fit(dataset)
        # kmeans.save_clusters("./out/" + str(k) + "_iris_kmeans.txt")
        sse_set[1].append(kmeans.inertia_)

        def sse_cluster(indicies, n_clusters):
            centroids = [np.zeros(np.shape(dataset[0]))] * n_clusters
            for index in range(len(indicies)):
                centroids[indicies[index]] += dataset[index]
            sse_values = [0] * n_clusters
            for index in range(len(indicies)):
                sse_values[indicies[index]] += _euclidian(centroids[indicies[index]], dataset[index]) ** 2
            return sum(sse_values)


        hac_single = Sk_HAC(n_clusters=k, linkage='single')
        hac_single.fit(dataset)
        # hac_single.save_clusters("./out/" + str(k) _ "_hac_single.txt")
        sse_set[2].append(sse_cluster(hac_single.labels_, hac_single.n_clusters))

        hac_complete = Sk_HAC(n_clusters=k, linkage='complete')
        hac_complete.fit(dataset)
        # hac_complete.save_clusters("./out/" + str(k) _ "_hac_complete.txt")
        sse_set[3].append(hac_complete.labels_, hac_complete.n_clusters)

    
    sse_with_label = [[],[],[],[]]

    for k in range(2, 8):
        run_both(k, ldata, sse_with_label)

    np.savetxt('./out/sk_iris_with_label.txt', sse_with_label, delimiter=',')

# def other():









# basic_hac()
# basic_kmeans()
# debug()
# evaluation()
iris()