import numpy as np
import math

def generate_random_sequence():
    arr = np.random.uniform(0.0, 1.0, 50)
    diff = (np.sum(arr) - (0.5 * 50))
    args = np.argsort(arr)
    if diff > 0:
        arr[args[-26:-1]] -= diff / 25
    else:
        arr[args[:25]] -= diff / 25

    return arr


if __name__ == '__main__':
    for i in range(100):
        a = generate_random_sequence()
        print([l for l in a if l < 0 or l > 1])
        print(a.mean())