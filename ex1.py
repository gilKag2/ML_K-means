from scipy.misc import imread
from init_centroids import init_centroids
import numpy as np


def load():
    # data preperation (loading, normalizing, reshaping)
    path = 'dog.jpeg'
    A = imread(path)
    A = A.astype(float) / 255.
    img_size = A.shape
    X = A.reshape(img_size[0] * img_size[1], img_size[2])
    return X


def k_means(k):
    x = load()
    centroids = init_centroids(x, k)

    for count in range(10):
        print('iter %d' + ': '  + centroids)
        clusters = {}
        for i in centroids:
            clusters[i] = []
        for point in x:
            closest = find_closest_centroid(point, centroids)
            # add the point to the cluster group of the closest centroid.
            clusters[closest] += [point]
        update(centroids, clusters)



def find_closest_centroid(point, centroids):
    closest = centroids[0]
    curr_min_dist = 1000
    for i in centroids:
        dist = calc_dist(point, i)
        if dist < curr_min_dist:
            closest = i
    return closest


def update(centroids, clusters):
    for i in centroids:
        for j in i:
            # average value.
            centroids[i][j] = np.mean[clusters[j]]
    return centroids


def calc_dist(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)


def show():
    
