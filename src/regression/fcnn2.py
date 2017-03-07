import numpy as np
import sys

class FCNN(object):
    def __init__(self, i_n, o_n, h_n, h_l, func):
        self.ACT_FUNCS = {'sigmoid': self.sigmoid}
        self.ACT_FUNCS_D = {'sigmoid': self.sigmoid_der}

        if func not in self.ACT_FUNCS.keys():
            sys.exit("Unknown Activation Function: %s" % (func))

        if 0 in [i_n, o_n, h_n, h_l]:
            sys.exit("Net Topology Parameters can't be 0")

        print ("---Topology---")
        print (("Type: %s\nInput Dimension: %s\nOutput Dimension %s\nHidden "
               + "Layers: %s\nHidden Layer Neurons: %s\nActivation Function: %s")
               % ("Fully Connected", i_n, o_n, h_l, h_n, func))
        print ("--------------")

        # topology
        self.input_neurons = i_n
        self.output_neurons = o_n
        self.hidden_neurons = h_n
        self.hidden_layers = h_l
        self.activate = self.ACT_FUNCS[func]
        self.activate_der = self.ACT_FUNCS_D[func]

        # initialize weights randomly
        self.W_in = np.random.rand(self.input_neurons,
                                 self.hidden_neurons)

        self.W_out = np.random.rand(self.hidden_neurons,
                                 self.output_neurons)

        self.hidden_weights = []
        for layer in range(self.hidden_layers - 1):
            self.hidden_weights.append(np.random.rand(
                self.hidden_neurons, self.hidden_neurons))


    # 1 matrix row = 1 training example
    def forward(self, X):
        # propragating from input through 1st hidden layer
        prev_layer_output = self.activate(np.dot(X, self.W_in))

        # propragating through hidden layers
        for l_weights in self.hidden_weights:
            prev_layer_output = self.activate(np.dot(prev_layer_output, l_weights))

        # propragating from last hidden through output layer
        pred = self.activate(np.dot(prev_layer_output, self.W_out))

        return pred

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def sigmoid_der(self, z):
        pass

    def calc_error(self, predictions, y):
        pass

if __name__ == "__main__":
    nn = FCNN(2, 1, 3, 1, 'sigmoid')
    print (nn.forward(np.random.rand(2,2)))
