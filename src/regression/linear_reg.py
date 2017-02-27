import numpy as np

class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

def calc_sse(w, samples):
    error = 0
    for point in samples:
        error += (point.y - (np.dot(w, point.x))) ** 2

    return (0.5*error) / len(points)


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
    learning_rate = 0.01
    points = []
    points.append(Point([1,1], 5))
    points.append(Point([2,2], 9))
    points.append(Point([3,3], 13))
    points.append(Point([4,4], 17))
    points.append(Point([5,5], 21))
    points.append(Point([6,6], 25))

    w = [1.3, 2.9]
    b = 0.3

    #preprocessing
    for point in points:
        point.x = [1] + point.x

    w = [b] + w

    #print (calc_sse(w, points))

    #
    for i in range(100):
        w = update(w,points,learning_rate)
