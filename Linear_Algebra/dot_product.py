import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # handling mismatch
    if len(x) != len(y):
        raise ValueError("Length Mismatch")

    a = np.array(x)
    b = np.array(y)

    return np.dot(a, b)


def main():
    x = [1,2,3]
    y = [4,5,6]

    res = dot_product(x, y)
    print(res)

main()