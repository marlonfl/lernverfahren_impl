import numpy as np
import math
import sys
sys.path.append("../util/")
import get_mnist
import quickselect as q
from scipy.spatial.distance import euclidean

class kNN(object):
    set_size = 40
    def __init__(self, k, X_train, Y_train):
        self.k = k
        self.X_train = X_train
        self.Y_train = Y_train

        classes = {}


        for clazz in set(Y_train):
            assigned_samples = X_train[[x for x, label in enumerate(Y_train) if label == clazz]]
            classes[clazz] = np.mean(assigned_samples, axis=0)

        self.centroids = classes

    def predict(self, X_test):
        predictions = []
        knc = 0
        knn = 0
        for sample in X_test:
            distances = {clazz: euclidean(sample, centroid) for clazz, centroid in self.centroids.items()}
            vals = list(distances.values())
            v = np.array(vals).argsort()
            if distances[v[1]] - distances[v[0]] < distances[v[0]]/30:
                predictions.append(self.pred_knn(sample))
                knn += 1
            else:
                predictions.append(min(distances, key=distances.get))
                knc += 1

        print ("\n Centroids: " + str(knc))
        print ("Neighbors: " + str(knn))
        return predictions

    def pred_knn(self, sample):
        distances = [euclidean(x, sample) for x in self.X_train]
        tmp_dst = distances[:]
        k_smallest = q.select(tmp_dst, 0, len(tmp_dst) -1, self.k)
        labels = [self.Y_train[i] for i, dist in enumerate(distances)
                                                        if dist < k_smallest]
        return max(set(labels), key=labels.count)

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
        k = 10

    num_dim = 200
    X_train, Y_train, X_test, Y_test = get_mnist.load_mnist()

    X_train = X_train[0:3000]
    Y_train = Y_train[0:3000]
    X_test = X_test[0:500]
    Y_test = Y_test[0:500]

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
