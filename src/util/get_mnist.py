from mnist import MNIST
import os
import sys


def load_mnist():
    if not os.path.exists("../../mnist/t10k-images-idx3-ubyte"):
        sys.exit("MNIST data not downloaded. Put it in the mnist-folder. \n"
            + "https://pypi.python.org/pypi/python-mnist/ <-- Tutorial")

    print ("Loading MNIST data...", end="", flush=True)
    mnist_data = MNIST("../../mnist")
    train_imgs, train_labels = mnist_data.load_training()
    test_imgs, test_labels = mnist_data.load_testing()
    print ("Done")

    return train_imgs, train_labels, test_imgs, test_labels
