import numpy as np
import math
import random
import matplotlib.pyplot as plt

def init_centers(X_train, k):
    dim_limits = []
    centers = []
    for i in range(len(X_train[0])):
        fvals = X_train[:,i]
        dim_limits.append((min(fvals), max(fvals)))

    for k_i in range(k):
        center = []
        for mini, maxi in dim_limits:
            center.append(random.uniform(mini,maxi))
        centers.append(center)

    return centers

def train(X_train, k, iterations):
    centers = init_centers(X_train, k)
    print ("initial centers " + str(centers))
    for i in range(iterations):
        assignments = [[]] * k
        for sample in X_train:
            assignments[assign_to_center(centers, sample)].append(sample)

        centers = update_centers(assignments)

    return centers

def update_centers(assignments):
    centers = []
    for i in range(len(assignments)):
        center = []
        for dim in range(len(assignments[0][0])):
            center.append(np.mean(np.array(assignments)[i][:,dim]))
        centers.append(center)
    print ("Centers updated to: " + str(centers))
    return centers

def assign_to_center(centers, sample):
    dists = []
    for center in centers:
        dists.append(euclid(center, sample))
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


    train_cl1 = np.array([np.array([random.uniform(0, 0.45), random.uniform(0, 0.6)]) for i in range(100)])
    train_cl2  = np.array([np.array([random.uniform(0.55, 1), random.uniform(0.4, 1)]) for i in range(100)])
    train_data = np.concatenate((train_cl1, train_cl2), axis=0)
    centers = train(train_data, 3, 10)
    # print (centers)
    #
    #plt.scatter(*zip(*train_cl1),color='red')
    #plt.scatter(*zip(*train_cl2),color='blue')
    plt.scatter(*zip(*centers), color='green', s=200)
    plt.ylabel('some numbers')
    plt.show()
