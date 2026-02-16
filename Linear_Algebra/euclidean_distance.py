import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    a = np.array(x)
    b = np.array(y)

    res = np.subtract(a, b)
    res = np.power(res, 2)
    res = np.sum(res)
    res = np.sqrt(res)
    return res


def main():
    x = [1,2,3]
    y = [4,5,6]

    res = euclidean_distance(x, y)
    print(res)

main()