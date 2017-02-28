import csv
import numpy as np
import sys
sys.path.append("../util/")
from make_sets import make_train_test
import math

class Naive_Bayes(object):
    class_gaussians = {}
    class_priori = {}

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.train()

    def train(self):
        n_samples = {}
        for clazz in set(self.Y):
            cl_samples = self.X[[i for i, label in enumerate(self.Y)
                                                if label == clazz]]
            n_samples[clazz] = len(cl_samples)

            means = np.mean(cl_samples, axis=0)
            st_devs = np.std(cl_samples, axis=0)
            self.class_gaussians[clazz] = zipp(means, st_devs)

        for clazz in n_samples.keys():
            self.class_priori[clazz] = n_samples[clazz] / sum(n_samples.values())

    def predict(self, X_test):
        predictions = []
        for sample in X_test:
            probs = {}
            for clazz in self.class_gaussians.keys():
                pdf_params = self.class_gaussians[clazz]
                feat_probs = []
                for i, params in enumerate(pdf_params):
                    mean = params[0]
                    st_dev = params[1]
                    exponent = math.exp(-(math.pow(sample[i]-mean,2)/(2*math.pow(st_dev,2))))
                    feat_probs.append((1 / (math.sqrt(2*math.pi) * st_dev)) * exponent)

                probs[clazz] = np.prod(feat_probs) * self.class_priori[clazz]

            predictions.append(max(probs, key=probs.get))

        return predictions

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

def zipp(a,b):
    result = []
    for i in range(len(a)):
        result.append((a[i], b[i]))

    return np.array(result)

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_data(
                        "../../diabetes/pima-indians-diabetes.csv")

    nb = Naive_Bayes(X_train, Y_train)
    predictions = nb.predict(X_test)

    print ("Accuracy: " + str(eval(predictions, Y_test)))
