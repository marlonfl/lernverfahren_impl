import numpy as np
import math
import sys
sys.path.append("../util/")
import get_mnist
from scipy.spatial.distance import euclidean

class kNN(object):
    def __init__(self, k, X_train, Y_train):
        self.k = k
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test):
        predictions = []
        for sample in X_test:
            distances = [euclidean(x, sample) for x in self.X_train]
            indices = np.array(distances).argsort()[:self.k]
            labels = [Y_train[index] for index in indices]
            predictions.append(max(set(labels), key=labels.count))

        return predictions

# returns accuracy
def eval(preds, labels):
    correct = 0
    for i, prediction in enumerate(preds):
        if prediction == labels[i]:
            correct += 1

    return correct/len(labels)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        k = int(sys.argv[1])
    else:
        k = 5

    X_train, Y_train, X_test, Y_test = get_mnist.load_mnist()

    X_train = X_train[0:500]
    Y_train = Y_train[0:500]
    X_test = X_test[0:100]
    Y_test = Y_test[0:100]

    classifier = kNN(k, X_train, Y_train)
    predictions = classifier.predict(X_test)
    err_rate = 1 - eval(predictions, Y_test)

    print ("Error Rate: " + str(err_rate*100) + "%")
