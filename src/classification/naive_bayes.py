import csv
import numpy as np
import sys
sys.path.append("../util/")
from make_sets import make_train_test

class Naive_Bayes(object):

    means = []
    st_devs = []

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        train()

    def train(self):
        pass

    def predict(self):
        pass

def load_data(fn):
    data = np.array((list(csv.reader(open(fn, 'r')))))
    labels = data[:,-1].astype(np.int)
    observations = np.array([sample[:-1] for sample in data]).astype(np.float)
    scale(observations)

    return make_train_test(observations, labels, 4)

def scale(X):
    minima = np.amin(X, axis=0)
    maxima = np.amax(X, axis=0)
    for dim in range(len(X[0])):
        for x in X:
            if maxima[dim] == minima[dim]:
                x[dim] = 0
            else:
                x[dim] = (x[dim] - minima[dim]) / (maxima[dim] - minima[dim])

# returns accuracy
def eval(Y_pred, Y_test):
    correct = 0
    for i, prediction in enumerate(Y_pred):
        if prediction == Y_test[i]:
            correct += 1

    return correct/len(X_test)


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_data(
                        "../../diabetes/pima-indians-diabetes.csv")

    nb = Naive_Bayes(X_train, Y_train)
