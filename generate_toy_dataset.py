import numpy as np

def generate_random_sequence():
    arr = np.random.uniform(0.0, 1.0, 50)
    diff = (np.sum(arr) - (0.5 * 50))
    args = np.argsort(arr)
    if diff > 0:
        arr[args[-26:-1]] -= diff / 25
    else:
        arr[args[:25]] -= diff / 25

    return arr


def create_toy_data(n_seqs=10000):
    dataset = []
    for i in range(n_seqs):
        dataset.append(generate_random_sequence())
    return np.expand_dims(dataset, 2)


if __name__ == '__main__':
    dataset = create_toy_data()
    print('done')