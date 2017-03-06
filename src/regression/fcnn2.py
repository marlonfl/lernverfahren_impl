import numpy as np

class Fully_Connected_NN(object):
    def __init__(self):
        self.input_neurons = 2
        self.output_neurons = 1
        self.hidden_layers = 1
        self.hidden_neurons = 3

        self.W1 = np.random.rand(self.input_neurons,
                                 self.hidden_neurons)

        self.W2 = np.random.rand(self.hidden_neurons,
                                 self.output_neurons)

    # 1 matrix row = 1 training example
    def forward(self, X):
        # sum for every neuron for every input in 1st hidden layer
        self.z2 = np.dot(X, self.W1)
        # activation on every neuron for every input
        self.a2 = self.sigmoid(self.z2)

        self.z3 = np.dot(X, self.a2)
        Y_pred = self.sigmoid(self.z3)

        return Y_pred

    def sigmoid(self, z):
        return 1/1(np.exp(-z))

    def calc_error(self, predictions, y):

if __name__ == "__main__":
    nn = Fully_Connected_NN()
    print (nn.forward())
