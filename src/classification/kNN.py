import numpy as np
import math
import sys
sys.path.append("../util/")
import get_mnist
import quickselect as q
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
            # indices = np.array(distances).argsort()[:self.k]
            # labels = [Y_train[index] for index in indices]
            tmp_dst = distances[:]
            k_smallest = q.select(tmp_dst, 0, len(tmp_dst) -1, self.k)
            labels = [self.Y_train[i] for i, dist in enumerate(distances)
                                                        if dist < k_smallest]
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
        k = 12

    num_dim = 150
    X_train, Y_train, X_test, Y_test = get_mnist.load_mnist()

    X_train = X_train[0:3000]
    Y_train = Y_train[0:3000]
    X_test = X_test[0:300]
    Y_test = Y_test[0:300]

    dim = len(X_train[0])

    print ("Reducing Dimensionality...")
    print (str(dim) + " -> " + str(num_dim))
    variances = np.var(X_train, axis=0)

    indices = variances.argsort()[:len(X_train[0])-num_dim]

    X_train = np.delete(X_train, indices, axis=1)
    X_test = np.delete(X_test, indices, axis=1)

    print ("Predicting test instances...", end="", flush=True)
    classifier = kNN(k, X_train, Y_train)
    predictions = classifier.predict(X_test)
    print ("Done")
    err_rate = 1 - eval(predictions, Y_test)

    print ("-----------")
    print ("Error Rate: " + str(err_rate*100) + "%")
