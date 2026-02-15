import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """

    if sum(a) == 0 or sum(b) == 0:
        return 0.0
    dot = np.dot(a, b)
    l2_a = ecludian_norm(a)
    l2_b = ecludian_norm(b)
    res = dot/(l2_a * l2_b)
    return res
    

def ecludian_norm(a):
    sum = 0
    for n in a:
        sum += np.power(n, 2)

    return np.sqrt(sum)


def main():
    # Test case 
    a = [0, 0, 0] 
    b = [1, 2, 3]

    res = cosine_similarity(a, b)
    print(res)

main()