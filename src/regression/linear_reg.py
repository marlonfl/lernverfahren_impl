import numpy as np
import matplotlib.pyplot as plt

class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

def calc_sse(w, samples):
    error = 0
    for point in samples:
        error += (point.y - (np.dot(w, point.x))) ** 2

    return (0.5*error) / len(points)

# batch gradient descent
def update(w_old, samples, alpha):
    print ("w: " + str(w_old[1:]))
    print ("b: " + str(w_old[0]))
    print (calc_sse(w_old, samples))

    w = [0] * len(w_old)
    for j in range(len(w_old)):
        p_d = sum([(point.x[j] * (np.dot(w_old, point.x) - point.y))
                                                         for point in samples])
        w[j] = w_old[j] - (alpha * p_d)

    return w


if __name__ == "__main__":
    learning_rate = 0.003
    points = []
    points.append(Point([1], 5))
    points.append(Point([2], 9))
    points.append(Point([3], 13))
    points.append(Point([4], 17))
    points.append(Point([5], 21))
    points.append(Point([6], 25))

    w = [1.3]
    b = 0.3

    #preprocessing
    for point in points:
        point.x = [1] + point.x

    w = [b] + w

    #print (calc_sse(w, points))

    m = w[1]
    b = w[0]
    asdf = []
    for x in range(0, points[-1].x[1] + 1):
        asdf.append(m*x + b)
    plt.plot(asdf)

    for i in range(20):
        w = update(w,points,learning_rate)
        m = w[1]
        b = w[0]
        asdf = []
        for x in range(0, points[-1].x[1] + 1):
            asdf.append(m*x + b)
        plt.plot(asdf)

    p = []
    for point in points:
        p.append([point.x[1], point.y])

    plt.scatter(*zip(*p),color='red')
    plt.show()
