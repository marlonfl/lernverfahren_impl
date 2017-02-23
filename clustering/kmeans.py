import numpy as np
import math
import random
import matplotlib.pyplot as plt


class Center(object):
    coordinates = np.array([])

    def __init__(self, dimensions):
        self.coordinates = np.array([random.uniform(0, 1)
                                        for i in range(dimensions)])
    def update_position(self, assigned):
        self.coordinates = np.mean(assigned, axis=0)


def init_centers(k, dim):
    return [Center(dim) for i in range(k)]

def train(X_train, k, iterations):
    centers = init_centers(k, len(X_train[0]))

    for i in range(iterations):
        assignments = [[] for i in range(k)]
        for sample in X_train:
            a_c = assign_to_center(centers, sample)
            assignments[a_c].append(sample)

        for i, assigned in enumerate(assignments):
            centers[i].update_position(assigned)

    return centers

def assign_to_center(centers, sample):
    dists = []
    for center in centers:
        dists.append(euclid(center.coordinates, sample))

    return dists.index(min(dists))

def predict():
    pass

def eval(labels, predictions):
    pass

def load_model():
    pass

def load_files(path):
    pass

def load_unsupervised(path):
    pass

def save_model():
    pass

def euclid(a,b):
    dist = 0
    for i, feat in enumerate(a):
        dist += (a[i] - b[i]) ** 2

    return math.sqrt(dist)


if __name__ == "__main__":
    # k = sys.argv[1]
    # iterations = sys.argv[2]
    # X_train = load_unsupervised(sys.argv[3])
    # X_test = load_unsupervised(sys.argv[4])


    train_cl1 = np.array([np.array([random.uniform(0, 0.45), random.uniform(0, 0.6)]) for i in range(20)])
    train_cl2  = np.array([np.array([random.uniform(0.55, 1), random.uniform(0.4, 1)]) for i in range(20)])
    train_data = np.concatenate((train_cl1, train_cl2), axis=0)
    centers = train(train_data, 2, 10)
    for c in centers:
        print (c.coordinates)

    c_coords = [c.coordinates for c in centers]
    plt.scatter(*zip(*train_cl1),color='red')
    plt.scatter(*zip(*train_cl2),color='blue')
    plt.scatter(*zip(*c_coords), color='green', s=200)
    plt.ylabel('some numbers')
    plt.show()
