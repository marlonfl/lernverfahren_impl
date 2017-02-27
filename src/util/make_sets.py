import numpy as np
import random

def make_train_test(X, Y, split):
    print ("Creating validation and test sets...")
    all_indices = range(len(X))
    test_indices = random.sample(all_indices, int(len(X)/split))
    training_indices = [i for i in all_indices if i not in test_indices]
    X_test = np.array([X[i] for i in test_indices])
    Y_test = np.array([Y[i] for i in test_indices])
    X_train = np.array([X[i] for i in training_indices])
    Y_train = np.array([Y[i] for i in training_indices])
    print ("Successfully created sets (Test: %d | Training: %d)"
                % (len(X_test), len(X_train)))
    return X_train, Y_train, X_test, Y_train
