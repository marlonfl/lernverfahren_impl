import numpy as np
import math
import sys

class kNN(object):
    def __init__(self, k, X_train, Y_train):
        self.k = k
        self.X_train = X_train
        self.Y_train = Y_train

    def predict():
        pass

def load_samples(path):
    return ([],[])

def euclid(a,b):
    dist = 0
    for i, feat in enumerate(a):
        dist += (a[i] - b[i]) ** 2

    return math.sqrt(dist)

if __name__ == "__main__":
    k = int(sys.argv[1])
    X_train, Y_train = load_samples(sys.argv[2])
    X_test, Y_test = load_samples(sys.argv[3])

    kNN = kNN(k, X_train, Y_train)
