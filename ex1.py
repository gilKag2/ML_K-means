######################
# Gil Kagan
# 315233221
#######################

from scipy.misc import imread
import numpy as np


# initiate the centroids.
def init_centroids(X, K):
    """
    Initializes K centroids that are to be used in K-Means on the dataset X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    K : int
        The number of centroids.

    Returns
    -------
    centroids : ndarray, shape (K, n_features)
    """
    if K == 2:
        return np.asarray([[0., 0., 0.],
                           [0.07843137, 0.06666667, 0.09411765]])
    elif K == 4:
        return np.asarray([[0.72156863, 0.64313725, 0.54901961],
                           [0.49019608, 0.41960784, 0.33333333],
                           [0.02745098, 0., 0.],
                           [0.17254902, 0.16862745, 0.18823529]])
    elif K == 8:
        return np.asarray([[0.01568627, 0.01176471, 0.03529412],
                           [0.14509804, 0.12156863, 0.12941176],
                           [0.4745098, 0.40784314, 0.32941176],
                           [0.00784314, 0.00392157, 0.02745098],
                           [0.50588235, 0.43529412, 0.34117647],
                           [0.09411765, 0.09019608, 0.11372549],
                           [0.54509804, 0.45882353, 0.36470588],
                           [0.44705882, 0.37647059, 0.29019608]])
    elif K == 16:
        return np.asarray([[0.61568627, 0.56078431, 0.45882353],
                           [0.4745098, 0.38039216, 0.33333333],
                           [0.65882353, 0.57647059, 0.49411765],
                           [0.08235294, 0.07843137, 0.10196078],
                           [0.06666667, 0.03529412, 0.02352941],
                           [0.08235294, 0.07843137, 0.09803922],
                           [0.0745098, 0.07058824, 0.09411765],
                           [0.01960784, 0.01960784, 0.02745098],
                           [0.00784314, 0.00784314, 0.01568627],
                           [0.8627451, 0.78039216, 0.69803922],
                           [0.60784314, 0.52156863, 0.42745098],
                           [0.01960784, 0.01176471, 0.02352941],
                           [0.78431373, 0.69803922, 0.60392157],
                           [0.30196078, 0.21568627, 0.1254902],
                           [0.30588235, 0.2627451, 0.24705882],
                           [0.65490196, 0.61176471, 0.50196078]])
    else:
        print('This value of K is not supported.')
        return None


# loads the image.
def load():
    # data preperation (loading, normalizing, reshaping)
    path = 'dog.jpeg'
    A = imread(path)
    A = A.astype(float) / 255.
    img_size = A.shape
    X = A.reshape(img_size[0] * img_size[1], img_size[2])
    return X


# calculates the average position for the new centroid,
# in the x, y and z axes.
def calc_avg_location(points):
    avg_x = 0
    avg_y = 0
    avg_z = 0
    for point in points:
        avg_x += point[0]
        avg_y += point[1]
        avg_z += point[2]
    size = len(points)
    if size:
        avg_x /= size
        avg_y /= size
        avg_z /= size
    return [avg_x, avg_y, avg_z]


# update the centroids positions to be the the average of the points in the cluster.
def update(centroids, cluster):
    for i in range(len(centroids)):
        centroids[i] = calc_avg_location(cluster[i])
    return centroids


# calculates the norm, to the power of 2 (helps with negative values).
def calc_dist(point1, point2):
    return np.linalg.norm(point1 - point2) ** 2


# find closest centroid to the point. returns array of 2 results:
# the index of the centroid, and the distance.
def find_closest_centroid_idx_and_distance(point, centroids):
    closest_idx = 0
    counter = 0
    curr_min_dist = 1000
    for i in centroids:
        dist = calc_dist(point, i)
        if dist < curr_min_dist:
            closest_idx = counter
            curr_min_dist = dist
        counter += 1
    # returns the index and the distance in an array.
    return [closest_idx, curr_min_dist]


def print_cent(cent):
    if type(cent) == list:
        cent = np.asarray(cent)
    if len(cent.shape) == 1:
        return ' '.join(str(np.floor(100 * cent) / 100).split()). \
            replace('[ ', '[').replace('\n', ' ').replace(' ]',
                                                          ']').replace(
            ' ', ', ')
    else:
        return ' '.join(str(np.floor(100 * cent) / 100).split()). \
                   replace('[ ', '[').replace('\n', ' ').replace(' ]',
                                                                 ']').replace(
            ' ', ', ')[1:-1]

# the algorithm.
def k_means(k):
    x = load()
    centroids = init_centroids(x, k)
    total_loss = []
    print('k=%d:' % k)
    # do 11 iterations of the algorithm, from 0 to 10.
    for count in range(11):
        print('iter %d: %s' % (count, print_cent(centroids)))
        clusters = {key: [] for key in range(k)}
        loss = 0
        for point in x:
            # holds the index in the first pos, and distance in the second.
            index_and_dist = find_closest_centroid_idx_and_distance(point, centroids)
            # add the point to the cluster group of the closest centroid.
            clusters[index_and_dist[0]] += [point]
            loss += index_and_dist[1]
        # update the centroids location according to the mean value.
        update(centroids, clusters)
        # get average of the loss
        loss /= len(x)
        loss = np.floor(loss * 10000) / 10000
        total_loss.append(loss)


if __name__ == '__main__':
    for num_of_clusters in [2, 4, 8, 16]:
        k_means(num_of_clusters)
